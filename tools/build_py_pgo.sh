#!/bin/sh
# Build the Python package guided by a recorded run (profile-guided optimisation): an instrumented build, a practice
# run on made-up data (tools/pgo_workload.py), then the real build using what the run recorded. Results are the same
# as a plain build; only speed changes. Needs `rustup component add llvm-tools-preview`.
set -e
cd "$(dirname "$0")/.."
PROF=$(mktemp -d)
LLVM="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
# .cargo/config.toml's flags are replaced by RUSTFLAGS, so the chip's own instructions are asked for again here
RUSTFLAGS="-C target-cpu=native -C profile-generate=$PROF" tools/build_py.sh
.venv/bin/python -I tools/pgo_workload.py
"$LLVM/llvm-profdata" merge -o "$PROF/merged.profdata" "$PROF"
RUSTFLAGS="-C target-cpu=native -C profile-use=$PROF/merged.profdata" tools/build_py.sh
rm -rf "$PROF"
