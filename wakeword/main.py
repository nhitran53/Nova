"""Wake-word listener CLI.

--handoff  always listen for the wake word; on detection start the router/voice_server conversation and release the mic
--dry-run  always listen and just print detections (no voice_server, no API key needed)
(neither)  standalone: after the wake word, stream the mic to ElevenLabs Scribe and print the transcript
"""
import argparse
import contextlib
import logging
import sys
from pathlib import Path

from config import PLACEHOLDER_MODEL, ConfigError, WakeConfig, load_api_key
from detector import WakeDetector, ensure_feature_models, find_model_files
from handoff import VoiceServerClient, WakeService
from listener import Listener, MicFrames, WavFrames
from stt import MODES, RedactingFilter, SilenceEndpointer, Transcriber


def build_parser(defaults):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--placeholder", action="store_true",
                   help=f"PLACEHOLDER: use the pretrained '{PLACEHOLDER_MODEL}' model instead of custom hey_nova/nova models")
    p.add_argument("--models-dir", type=Path, default=defaults.models_dir, help="folder of custom .onnx models (default: %(default)s)")
    p.add_argument("--model", action="append", default=[], metavar="PATH",
                   help="explicit .onnx model path (repeatable). Default: every .onnx in --models-dir (normally models/hey_nova.onnx)")
    p.add_argument("--threshold", type=float, default=defaults.threshold, help="detection score threshold (default: %(default)s)")
    p.add_argument("--cooldown", type=float, default=defaults.cooldown_s, help="seconds to ignore detections after one (default: %(default)s)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--handoff", action="store_true",
                      help="on a wake word, start the voice_server conversation (POST /voice/start), release the mic until it ends, resume")
    mode.add_argument("--dry-run", "--no-handoff", dest="dry_run", action="store_true",
                      help="only print detections: no voice_server, no STT, no API key")
    p.add_argument("--voice-url", default=defaults.voice_url, help="router/voice_server base URL (default: %(default)s)")
    p.add_argument("--conversation-timeout", type=float, default=300.0,
                   help="--handoff: ask voice_server to stop a conversation that lasts longer than this many seconds (default: %(default)s)")
    p.add_argument("--handoff-cooldown", type=float, default=3.0,
                   help="--handoff: seconds to wait after a conversation before the mic reopens (default: %(default)s)")
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
        return None, (f"no wake-word model found: drop hey_nova.onnx into {args.models_dir}{'/' if not str(args.models_dir).endswith('/') else ''} "
                      "(or pass --model PATH; --placeholder uses a pretrained stand-in)")
    missing = [m for m in models if not Path(m).is_file()]
    if missing:
        return None, f"model file not found: {', '.join(missing)}"
    return models, None


def run_service(args, detector, device):
    """--handoff / --dry-run: the always-listening loop."""
    handoff = args.handoff
    if args.wav:
        frames = WavFrames(args.wav)  # one stream shared across mic "reopens", so the file is read once
        open_mic = lambda: contextlib.nullcontext(frames)
    else:
        open_mic = lambda: MicFrames(device=device, frame_timeout_s=5.0)

    def on_wake(detection):
        print(f"[wake] {detection.name} ({detection.score:.2f})" + (" - starting conversation" if handoff else " (dry run)"), flush=True)

    service = WakeService(open_mic, detector, VoiceServerClient(args.voice_url) if handoff else None, on_wake=on_wake,
                          cooldown_s=args.handoff_cooldown, conversation_timeout_s=args.conversation_timeout)
    print(f"listening for the wake word ({'handoff to ' + args.voice_url if handoff else 'dry run: detections only'}; Ctrl-C to quit)...",
          file=sys.stderr, flush=True)
    if handoff:
        service.check_server()
    try:
        service.run(args.max_wakes)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    return 0


def main(argv=None):
    args = build_parser(WakeConfig.from_env()).parse_args(argv)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    service_mode = args.handoff or args.dry_run
    api_key = None
    if not service_mode:  # standalone mode transcribes itself; handoff mode never needs the key (voice_server has its own)
        try:
            api_key = load_api_key()
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(RedactingFilter(api_key))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO if service_mode else logging.WARNING, handlers=[handler])
    logging.getLogger("websockets").setLevel(logging.WARNING)

    models, error = resolve_models(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        ensure_feature_models()  # first run only: needs internet once
    except Exception as exc:
        print(f"error: could not download openWakeWord's shared feature models (internet needed once): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        detector = WakeDetector(models, threshold=args.threshold, cooldown_s=args.cooldown)
    except Exception as exc:
        print(f"error: could not load wake-word model(s) {', '.join(map(str, models))}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if service_mode:
        return run_service(args, detector, device)
    transcriber = Transcriber(
        api_key, mode=args.stt_mode,
        endpointer_factory=lambda: SilenceEndpointer(silence_s=args.silence, rms_threshold=args.rms_threshold))

    def on_wake(detection):
        print(f"[wake] {detection.name} ({detection.score:.2f}) - listening...", flush=True)

    def on_transcript(text):
        print(f"[transcript] {text}" if text else "[transcript] (nothing heard)", flush=True)

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
