"""Runtime configuration: audio constants, env-driven defaults, and ELEVENLABS_API_KEY loading."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
KEY_VAR = "ELEVENLABS_API_KEY"
# The key lives in the router's .env (read-only here); NOVA_ENV_FILE or the environment can override it.
DEFAULT_ENV_FILE = HERE.parent / "router" / ".env"

SAMPLE_RATE = 16000          # openWakeWord and Scribe both take 16 kHz mono int16
FRAME_SAMPLES = 1280         # 80 ms at 16 kHz: the chunk size openWakeWord is tuned for
DEFAULT_VOICE_URL = "http://127.0.0.1:8765"  # router/voice_server.py default
PLACEHOLDER_MODEL = "hey_jarvis"  # pretrained stand-in until custom hey_nova / nova models exist


class ConfigError(RuntimeError):
    pass


def load_api_key(env_file=None):
    """Return the ElevenLabs key from the environment, else from the .env file. Never logs or echoes it."""
    key = os.environ.get(KEY_VAR, "").strip()
    if key:
        return key
    path = Path(env_file or os.environ.get("NOVA_ENV_FILE") or DEFAULT_ENV_FILE)
    if path.is_file():
        key = (dotenv_values(path).get(KEY_VAR) or "").strip()
    if not key:
        raise ConfigError(f"{KEY_VAR} is not set: export it or put it in {path}")
    return key


@dataclass(frozen=True)
class WakeConfig:
    models_dir: Path
    threshold: float
    cooldown_s: float
    voice_url: str

    @classmethod
    def from_env(cls):
        return cls(
            models_dir=Path(os.environ.get("WAKEWORD_MODELS_DIR", HERE / "models")),
            threshold=float(os.environ.get("WAKEWORD_THRESHOLD", "0.3")),
            cooldown_s=float(os.environ.get("WAKEWORD_COOLDOWN_S", "2.0")),
            voice_url=os.environ.get("WAKEWORD_VOICE_URL", DEFAULT_VOICE_URL),
        )
