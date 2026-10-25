#!/usr/bin/env bash
# One-command demo start: always-listening wake word that hands off to router/voice_server.
# Usage: ./run.sh [extra main.py flags, e.g. --device 3 --threshold 0.4]
cd "$(dirname "$0")" && exec uv run python main.py --handoff "$@"
