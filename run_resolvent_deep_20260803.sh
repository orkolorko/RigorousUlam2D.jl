#!/bin/bash
# Deep resolvent certification push, Lorenz — 2026-08-03
# Targets the gate depths from the log: a=0.2 needs d*=7, a=0.1 needs d*=9.
# a=0.3 is re-run at depth 7 purely as a convergence check against the depth-6 value (R1=11.8).
cd /home/isaia/Code/RigorousUlam2D.jl || exit 1
export JULIA_NUM_THREADS=16

run_stage () {
  depth=$1; noises=$2; out="results/resolvent_circle_around1_lorenz_d${depth}.txt"
  echo "################################################################"
  echo "### STAGE depth=$depth noises=$noises  START $(date -Is)"
  echo "################################################################"
  start=$(date +%s)
  CACHED=1 DEPTH=$depth MAP=lorenz NOISES=$noises RCIRC=0.6 \
    julia --project=examples examples/17_reduced_resolvent.jl 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  end=$(date +%s)
  echo "### STAGE depth=$depth DONE rc=$rc elapsed=$(( (end-start)/60 ))m$(( (end-start)%60 ))s  $(date -Is)"
  echo "elapsed_seconds=$(( end-start )) rc=$rc" >> "$out"
}

run_stage 7 "0.3,0.2"
run_stage 9 "0.1"

echo "ALL STAGES COMPLETE $(date -Is)"
