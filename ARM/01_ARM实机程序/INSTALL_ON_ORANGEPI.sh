#!/bin/sh
set -eu
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python3 verify_package.py
exec python3 launcher/install_shortcuts.py --restore-if-missing
