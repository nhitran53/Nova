"""Wake-word listener CLI: say the wake word, then speak; the ElevenLabs Scribe transcript is printed."""
import argparse
import logging
import sys
from pathlib import Path

from config import PLACEHOLDER_MODEL, ConfigError, WakeConfig, load_api_key
from detector import WakeDetector, find_model_files
from listener import Listener, MicFrames, WavFrames
from stt import MODES, RedactingFilter, SilenceEndpointer, Transcriber


def build_parser(defaults):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--placeholder", action="store_true",
                   help=f"PLACEHOLDER: use the pretrained '{PLACEHOLDER_MODEL}' model instead of custom hey_nova/nova models")
    p.add_argument("--models-dir", type=Path, default=defaults.models_dir, help="folder of custom .onnx models (default: %(default)s)")
    p.add_argument("--model", action="append", default=[], metavar="PATH", help="explicit .onnx model path (repeatable; overrides --models-dir)")
    p.add_argument("--threshold", type=float, default=defaults.threshold, help="detection score threshold (default: %(default)s)")
    p.add_argument("--cooldown", type=float, default=defaults.cooldown_s, help="seconds to ignore detections after one (default: %(default)s)")
    p.add_argument("--stt-mode", choices=MODES, default="auto", help="auto = realtime, falling back to batch (default)")
    p.add_argument("--silence", type=float, default=1.5, help="seconds of silence that end an utterance (default: %(default)s)")
    p.add_argument("--rms-threshold", type=float, default=400.0, help="int16 RMS above which a frame counts as speech (default: %(default)s)")
    p.add_argument("--device", help="sounddevice input device (index or name); default mic if omitted")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--wav", type=Path, help="read 16 kHz mono 16-bit WAV instead of the mic (smoke testing)")
    p.add_argument("--max-wakes", type=int, help="exit after this many wake-ups")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_models(args):
    """Model paths/names to load, or an error string."""
    if args.placeholder:
        from openwakeword.utils import download_models
        download_models([PLACEHOLDER_MODEL])  # no-op for files already present
        print(f"*** PLACEHOLDER MODE: wake word is '{PLACEHOLDER_MODEL.replace('_', ' ')}', "
              "not 'hey nova' (train custom models: see README.md) ***", file=sys.stderr)
        return [PLACEHOLDER_MODEL], None
    models = args.model or find_model_files(args.models_dir)
    if not models:
        return None, f"no .onnx models in {args.models_dir}; train hey_nova/nova (see README.md) or pass --placeholder"
    return models, None


def main(argv=None):
    args = build_parser(WakeConfig.from_env()).parse_args(argv)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    try:
        api_key = load_api_key()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RedactingFilter(api_key))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, handlers=[handler])
    logging.getLogger("websockets").setLevel(logging.WARNING)

    models, error = resolve_models(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    detector = WakeDetector(models, threshold=args.threshold, cooldown_s=args.cooldown)
    transcriber = Transcriber(
        api_key, mode=args.stt_mode,
        endpointer_factory=lambda: SilenceEndpointer(silence_s=args.silence, rms_threshold=args.rms_threshold))

    def on_wake(detection):
        print(f"[wake] {detection.name} ({detection.score:.2f}) - listening...", flush=True)

    def on_transcript(text):
        print(f"[transcript] {text}" if text else "[transcript] (nothing heard)", flush=True)

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    try:
        if args.wav:
            Listener(WavFrames(args.wav), detector, transcriber, on_wake, on_transcript).run(args.max_wakes)
        else:
            with MicFrames(device=device) as mic:
                print("listening (Ctrl-C to quit)...", file=sys.stderr)
                Listener(mic, detector, transcriber, on_wake, on_transcript).run(args.max_wakes)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
