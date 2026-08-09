# Provenance header for results files. Source this, then call:
#
#   . scripts/provenance.sh
#   provenance_header "CACHED=1 DEPTH=7 MAP=lorenz ..." > results/foo.txt
#
# The point: a certified number is meaningless without the code that produced
# it. Emitting this from the runner makes provenance a byproduct of running
# rather than a discipline anyone has to remember -- which is what failed on
# 2026-08-03, when a killed run left result files with no record of anything.
provenance_header () {
    local dirty
    dirty=$(git status --porcelain 2>/dev/null | wc -l)
    echo "# ---------------- provenance ----------------"
    echo "# date         $(date -Is)"
    echo "# host         $(hostname)"
    echo "# commit       $(git rev-parse HEAD 2>/dev/null || echo '(not a git repo)')"
    echo "# branch       $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    if [ "$dirty" -gt 0 ]; then
        echo "# WORKTREE     DIRTY -- $dirty uncommitted path(s); numbers below are NOT reproducible from the commit above"
        git status --porcelain 2>/dev/null | sed 's/^/#              /'
    else
        echo "# worktree     clean"
    fi
    echo "# julia        $(julia --version 2>/dev/null)"
    echo "# threads      JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-unset}"
    echo "# nproc        $(nproc)"
    echo "# invocation   $*"
    echo "# --------------------------------------------"
}
