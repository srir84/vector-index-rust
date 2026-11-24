#!/bin/zsh
# Run the Rust tests, then each step's check in order, and print each step's summary line.
#   validate/run_all.sh            # steps 1 to 12, 14 to 21, then 23 to 26 (the held-back tests, steps 22 and 27,
#                                  # run on their own)
#   validate/run_all.sh 3 4 9      # just these
# Each step writes validate/results/stepN.json (every check with its numbers) and stepN.log (what it printed).
# Steps 11, 12, 18 to 21 and 24 to 26 time searches: run them on an otherwise idle machine.
# REUSE_GRID=1 reuses steps 18 and 21's saved grids of full-size builds (about an hour to make again).
set -u
cd "${0:A:h}/.."
if (( $# )); then steps=("$@"); else steps=({1..12} 14 15 16 17 18 19 20 21 23 24 25 26); fi
cargo test -p vindex --release 2>&1 | grep -E "^test result|FAILED|panicked"
for s in $steps; do
  .venv/bin/python -I validate/step$s.py > validate/results/step$s.log 2>&1
  print -r -- "$(grep -E "^step$s:" validate/results/step$s.log || print "step$s: did not finish (see validate/results/step$s.log)")"
done
