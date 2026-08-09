#!/usr/bin/env bash
# Generic resolvent runner: provenance in, index out.
#
#   ./run_resolvent.sh <map> <depth> <noises> [rcirc]
#   ./run_resolvent.sh lorenz 7 0.3,0.2
#
# ALWAYS launch long runs fully detached, or they die with the shell that
# started them (this is exactly how the 2026-08-03 push was lost):
#
#   setsid nohup ./run_resolvent.sh lorenz 7 0.3,0.2 > run_lorenz_d7.log 2>&1 < /dev/null &
#
# Verify with `ps -o pid,ppid,sid -p <pid>`: PPID must be 1 and the SID must
# differ from the launching shell's.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
. scripts/provenance.sh

map=${1:?usage: run_resolvent.sh <map> <depth> <noises> [rcirc]}
depth=${2:?}
noises=${3:?}
rcirc=${4:-0.6}

export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-16}
# The default output is keyed by map+depth, so re-running one noise of a
# partially-completed sweep would silently clobber the rows already earned.
# Set OUT to keep them:  OUT=results/..._d7_a02.txt ./run_resolvent.sh ...
out=${OUT:-results/resolvent_circle_around1_${map}_d${depth}.txt}
inv="CACHED=1 DEPTH=$depth MAP=$map NOISES=$noises RCIRC=$rcirc julia --project=examples examples/17_reduced_resolvent.jl"

echo "### START depth=$depth map=$map noises=$noises  $(date -Is)"
provenance_header "$inv" > "$out"

start=$(date +%s)
CACHED=1 DEPTH="$depth" MAP="$map" NOISES="$noises" RCIRC="$rcirc" \
    julia --project=examples examples/17_reduced_resolvent.jl 2>&1 | tee -a "$out"
rc=${PIPESTATUS[0]}
end=$(date +%s)

echo "elapsed_seconds=$(( end - start )) rc=$rc" >> "$out"
echo "### DONE rc=$rc elapsed=$(( (end-start)/60 ))m$(( (end-start)%60 ))s  $(date -Is)"

# Index only a run that actually succeeded. Indexing a killed run would put
# partial rows in the table with no marker that they are partial.
if [ "$rc" -eq 0 ]; then
    ./scripts/index_resolvent.sh "$out" "$map" "$depth"
else
    echo "rc=$rc -- NOT indexed. Inspect $out before trusting anything in it."
fi
exit "$rc"
