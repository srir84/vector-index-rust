#!/bin/sh
# Build the Python package (release) into .venv, then check that it loads.
set -e
cd "$(dirname "$0")/.."
(cd crates/vindex-py && VIRTUAL_ENV="$PWD/../../.venv" ../../.venv/bin/maturin develop --release -q 2>&1 | grep -v "^🛠\|^📦\|^✏️" || true)
.venv/bin/python -I -c "import vindex; print('vindex', vindex.__version__, 'built and loads')"
