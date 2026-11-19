#!/usr/bin/env sh
# Start Sports Helpers. The first run creates .venv and installs the packages.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Creating a virtual environment in .venv ..."
  python3 -m venv .venv
fi
exec .venv/bin/python run.py --yes "$@"
