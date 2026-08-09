#!/usr/bin/env bash
# Append the rows of a 17_reduced_resolvent.jl results file to results/index.jsonl,
# one JSON object per (noise, r) row, stamped with the commit that produced them.
#
#   scripts/index_resolvent.sh results/resolvent_circle_around1_lorenz_d7.txt lorenz 7
#
# Why: the deliverable of this project IS a table. Prose logs cannot answer
# "was this number before or after the periodic-boundary fix?" -- a commit-stamped
# index can, and the paper's gate table can be regenerated from it mechanically.
set -euo pipefail

file=${1:?usage: index_resolvent.sh <results-file> <map> <depth>}
map=${2:?}
depth=${3:?}
out=results/index.jsonl

# Normally the current commit/time are correct, because this is called by the
# runner immediately after the run. When BACKFILLING an old results file they
# are wrong -- and a silently wrong commit is worse than none -- so override:
#   INDEX_COMMIT=9b70fbe INDEX_TS=2026-08-03T13:17:00-03:00 scripts/index_resolvent.sh ...
commit=${INDEX_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
commit=$(git rev-parse "$commit" 2>/dev/null || echo "$commit")
dirty=$(( $(git status --porcelain 2>/dev/null | wc -l) > 0 ))
[ -n "${INDEX_COMMIT:-}" ] && dirty=0   # unknowable for a past run; do not guess
host=${INDEX_HOST:-$(hostname)}
stamp=${INDEX_TS:-$(date -Is)}

# A killed run may still have emitted complete rows before it died. Those rows
# are real and worth keeping -- what must never happen is their landing in the
# table indistinguishable from rows out of a run that finished. So they go in
# flagged, and only ever by hand:
#   INDEX_PARTIAL=1 scripts/index_resolvent.sh ...
partial=${INDEX_PARTIAL:-0}
[ "$partial" = 1 ] && partial=true || partial=false

# Data rows are the ones starting with a bare number: noise r |G22| s11 s22 floor bound.
# The bound column may be the literal "∞ (declines)", so take field 7 onward as text.
added=$(awk -v c="$commit" -v d="$dirty" -v h="$host" -v t="$stamp" \
            -v m="$map" -v dep="$depth" -v p="$partial" -v n=0 '
    /^[0-9]/ && NF >= 7 {
        bound = $7; for (i = 8; i <= NF; i++) bound = bound " " $i
        gsub(/"/, "", bound)
        finite = (bound ~ /^[0-9.eE+-]+$/) ? "true" : "false"
        printf "{\"ts\":\"%s\",\"host\":\"%s\",\"commit\":\"%s\",\"dirty\":%s,", t, h, c, (d ? "true" : "false")
        printf "\"map\":\"%s\",\"depth\":%s,\"n\":%d,\"partial\":%s,", m, dep, 4 ^ dep, p
        printf "\"noise\":%s,\"r\":%s,\"G22norm\":%s,\"sigma11\":%s,\"sigma22\":%s,\"floor\":%s,", $1, $2, $3, $4, $5, $6
        printf "\"bound_finite\":%s,\"bound\":\"%s\"}\n", finite, bound
        n++
    }
    END { print n > "/dev/stderr" }
' "$file" 2>&1 >>"$out" ) || true

# awk wrote rows to $out and the count to stderr, captured above.
echo "indexed ${added:-0} row(s) from $file into $out"
