from .base import BaseStrategy, Signal, SignalType
from .wt_adx_strategy import WTADXStrategy
from .ai_signal import AISignalStrategy
from .ai_expert_strategy import AIExpertStrategy
from .trend_confirm_strategy import TrendConfirmStrategy
from .sentinel_strategy import SentinelStrategy
from .sentinel_two_target import install_sentinel_two_target
from .sentinel_structure_v2 import install_sentinel_structure_v2
from .sentinel_log_clarity import install_sentinel_log_clarity
from .sentinel_quality_v3 import install_sentinel_quality_v3

# Sentinel-only lifecycle overlay:
# TP1 = +1.0R trim 60%, lock remaining SL at +0.3R, then keep the native
# mapped R1/S1 target or OPEN_SKY/OPEN_FLOOR dynamic runner as TP2.
install_sentinel_two_target(SentinelStrategy)

# Sentinel V2 architecture overlay:
# Engine A = existing 1H S/R reversal.
# Engine B = Sentinel X Fib 38.2-61.8 pullback + 15M trigger + MCDX.
# Probabilistic Target Engine is a soft factor / runner reference, not a hard gate.
install_sentinel_structure_v2(SentinelStrategy)

# Display-only diagnostics overlay.
install_sentinel_log_clarity(SentinelStrategy)

# Sentinel V3 final quality gate.  This runs LAST: both V2 entry engines must
# additionally agree with 4H direction, non-opposing 1H context and a 2/3 15M
# execution vote.  It also rolls back V1/V2 internal state when an entry is vetoed.
install_sentinel_quality_v3(SentinelStrategy)

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "WTADXStrategy",
    "AISignalStrategy",
    "AIExpertStrategy",
    "TrendConfirmStrategy",
    "SentinelStrategy",
]
