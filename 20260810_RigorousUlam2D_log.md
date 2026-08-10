# RigorousUlam2D — 2026-08-10

## Depth-7 Lorenz, a=0.2: the gate closes — finished 2026-08-09 11:35

Run: relaunched 2026-08-09 05:13, PID 24283, detached.
`setsid nohup env OUT=results/resolvent_circle_around1_lorenz_d7_a02.txt ./run_resolvent.sh lorenz 7 0.2 …`
Exited **rc=0**, elapsed 382 m 10 s (22930 s) on 16 threads. 3 rows indexed
(`partial: false`, commit `89c640b`, worktree dirty — flagged in the header).

Certified ‖(z−L)⁻¹‖ on |z−1|=r, Lorenz depth 7 (n=16384), a=0.2:

    r      R₁ ≤        R₁·δ_h     gate
    0.30   18.1256     0.7080     ✓
    0.20   18.0049     0.7033     ✓   ← min over r
    0.10   30.9969     1.2108     ✗

δ₇ = 1/128, δ_h = δ/a = 0.0390625.

**R₁ = 18.0049 (min over r) ⇒ R₁·δ_h = 0.7033 < 1. The gate closes.**
Abstract (mesh-independent) resolvent: R₁/(1−R₁δ_h) = **60.69**.

Convergence check: depth 6 at a=0.2 gave R₁ = 17.7577; depth 7 gives 18.0049,
**+1.39%**. R₁ is flat in depth while δ_h halves — which is exactly the mechanism
the gate relies on, and it is what the depth-6 table predicted (d\* = 7 for a=0.2:
the depth-6 gate was 1.387 ✗, depth 7 is 0.7033 ✓).

Gate table now certified at the required depth for two configurations:

    config              depth   R₁       R₁·δ_h   abstract R₁/(1−R₁δ_h)
    Lorenz a=0.3        6       11.8326  0.616    30.8
    Lorenz a=0.2        7       18.0049  0.7033   60.69

**Caveat:** this is the *certified reduced* resolvent bound, not the full Σ_res
error figure. Turning it into a ‖f − f_h‖ number still needs the BV bridge
(|M|^½ C_M var(ρ)) and a density at sufficient depth, and the (A)-vs-(B)
pipeline question is still open — see 20260803 log.

**Cost note:** a=0.2 at depth 7 took 6.4 h. Stage 2 (depth 9, a=0.1, n=262144)
is 16× the dimension. The old "plausibly an overnight run" estimate remains wrong;
re-scope before launching.
