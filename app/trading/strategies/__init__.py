from .base import BaseStrategy, Signal, SignalType
from .wt_adx_strategy import WTADXStrategy
from .ai_signal import AISignalStrategy
from .ai_expert_strategy import AIExpertStrategy
from .trend_confirm_strategy import TrendConfirmStrategy
from .sentinel_strategy import SentinelStrategy
from .sentinel_two_target import install_sentinel_two_target
from .sentinel_structure_v2 import install_sentinel_structure_v2
from .sentinel_log_clarity import install_sentinel_log_clarity

# Sentinel-only lifecycle overlay:
# TP1 = +1.0R trim 60%, lock remaining SL at +0.3R, then keep the native
# mapped R1/S1 target or OPEN_SKY/OPEN_FLOOR dynamic runner as TP2.
install_sentinel_two_target(SentinelStrategy)

# Sentinel V2 architecture overlay:
# Engine A = existing 1H S/R reversal.
# Engine B = Sentinel X Fib 38.2-61.8 pullback + 15M trigger + MCDX.
# Probabilistic Target Engine is a soft factor / runner reference, not a hard gate.
install_sentinel_structure_v2(SentinelStrategy)

# Display-only diagnostics overlay. Must run last so it can explain blockers from
# both Engine A and Engine B without changing any trading decision.
install_sentinel_log_clarity(SentinelStrategy)

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
