"""FlagMRL with pairwise nested trace ratio for the flag update only."""

from .flag_mrl import FlagMRLLoss


class FlagMRLTraceRatioLoss(FlagMRLLoss):
    """Share the entire encoder/MRL path; replace only the flag objective."""

    flag_objective = "trace_ratio"
