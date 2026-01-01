"""
Domain restriction utilities for efficient Ulam discretization.

This module has been largely superseded by build_adaptive_partition in partition_utils.jl.
Only minimal helper functions remain for legacy compatibility.
"""

using IntervalArithmetic, IntervalBoxes

# No exports - this module is deprecated in favor of build_adaptive_partition
