#!/usr/bin/env bash
# Lingora setup (Mac/Linux). Checks Python/Node/ffmpeg, creates .venv, installs deps.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Lingora SETUP ==="
echo

# 1. Python 3.12+
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.12+ from https://www.python.org/"
  exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
  echo "ERROR: Python 3.12+ required, found $PY_VER"
  exit 1
fi

# 2. Node 22+
if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node not found. Install Node.js 22+ from https://nodejs.org/"
  exit 1
fi
NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]")
if [ "$NODE_MAJOR" -lt 22 ]; then
  echo "ERROR: Node 22+ required, found $(node --version)"
  exit 1
fi
if ! command -v npx >/dev/null 2>&1; then
  echo "ERROR: npx not found. Reinstall Node.js with npm."
  exit 1
fi

# 3. ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH."
  echo "  macOS:   brew install ffmpeg"
  echo "  Ubuntu:  sudo apt install ffmpeg"
  exit 1
fi

# 4. .venv
if [ ! -d .venv ]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
fi

# 5. Install Python deps
echo "Installing Python dependencies ..."
# shellcheck source=/dev/null
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Hint first run
if [ ! -f channels/myvideo/.env ]; then
  echo
  echo "NOTE: No channel config found yet."
  echo "  Run ./RUN.sh once — the wizard creates channels/myvideo/.env for you."
  echo "  Then open it and paste GEMINI_API_KEYS (free: aistudio.google.com/apikey)."
fi

echo
echo "=== SETUP DONE ==="
echo "Run with: ./RUN.sh"
