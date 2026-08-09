# RigorousUlam2D experiment log — 2026-08-09

## 05:11 — depth-7 Lorenz resolvent: partial rows recovered, run died at reboot

Checked the run relaunched 2026-08-08 10:16 (`run_resolvent_deep_20260803.sh`,
PID 1543887, verified detached). It **did** produce data this time, then died.

Last write to `resolvent_deep_20260808.log` was 16:57:20. The machine was
**cleanly powered off at 22:06 and booted at 22:09** (`last -x reboot`;
journal of the previous boot shows an ordinary `systemd-poweroff`, not a crash
and not OOM). So the loss was a deliberate power cycle, not the harness this
time — and no amount of `setsid` survives a shutdown.

### Rows earned — Lorenz, depth 7 (n=16384), R=0.60, a=0.30

| noise | r | ‖G22‖ | σ₁₁(circle) | σ₂₂(circle) | floor | R₁ = ‖(z−L)⁻¹‖ ≤ |
|---|---|---|---|---|---|---|
| 0.300 | 0.30 | 0.0866 | 1.2761e-01 | 0.6134 | 8.1471e-02 | **12.2743** |
| 0.300 | 0.20 | 0.0866 | 8.9922e-02 | 0.7134 | 6.2703e-02 | 15.9481 |
| 0.300 | 0.10 | 0.0866 | 4.6087e-02 | 0.8134 | 3.4735e-02 | 28.7896 |

Depth-6 comparison at the same (a, r): 11.8326 / 15.4697 / 28.0288. The bound
is **essentially flat in depth** — it moved +3.7% at r=0.3 going 6→7, i.e. it
is not converging down with the grid. That is the expected behaviour (R₁ is a
property of the operator, not the discretisation) and it is what makes the gate
usable: R₁ stabilises while δ_h halves each level.

**a=0.2 never emitted a row.** Stage 1 spent roughly 7 h on the three a=0.3
rows, then 5 h on a=0.2 with no output before the power cut. Stage 2 (depth 9,
n=262144) was never reached.

### Caveats

- The three rows above are from a run that did **not** exit cleanly. They are
  complete rows and the header/gap data is intact, but they are indexed with
  `partial: true` in `results/index.jsonl` and must not be quoted as if they
  came from a finished sweep.
- The results file predates the provenance header, so its commit is recorded
  only in the index (backfilled as `7f3922e`, dirty state unknown).
- Cost estimate for what remains: a=0.2 at depth 7 did not finish in 5 h.
  Depth 9 is 16× the dimension of depth 7 — on this evidence stage 2 as
  written is not a run that completes overnight, and should be re-scoped
  before being launched again.

## 05:13 — relaunch, a=0.2 only

```
setsid nohup env OUT=results/resolvent_circle_around1_lorenz_d7_a02.txt \
    ./run_resolvent.sh lorenz 7 0.2 > run_lorenz_d7_a02_20260809.log 2>&1 < /dev/null &
```

PID 24283, own session and process group (SID = PGID = PID), distinct from the
launching shell — a group-directed kill cannot reach it. Julia up as 24305,
`ConsistentFPCSR = 1` confirmed on 16 threads.

Dropped a=0.3 from the invocation: those rows are already earned and re-running
them would have cost another seven hours. The `OUT` override exists so the
re-run cannot clobber the file holding them.

**Result:** running, no rows yet.

## Tooling committed (2e82dd1, 6e10b3f)

- `scripts/provenance.sh` — header (date, host, commit, **every dirty path**,
  julia version, threads, full invocation) written into each results file by
  the runner, so provenance is a byproduct of running rather than a discipline.
  Verified live: the header on the 05:13 run correctly flags the worktree dirty
  and names the three offending paths.
- `scripts/index_resolvent.sh` — commit-stamped rows into `results/index.jsonl`.
  `INDEX_COMMIT` / `INDEX_TS` / `INDEX_HOST` override for backfill (a silently
  wrong commit is worse than none); `INDEX_PARTIAL=1` flags salvaged rows.
- `run_resolvent.sh` — generic runner, `OUT`-overridable, indexes **only** on
  `rc=0`.

Index now holds 27 rows: lorenz d6 ×12 (`6560987`), henon d6 ×12 (`9b70fbe`),
lorenz d7 ×3 (`7f3922e`, partial).
