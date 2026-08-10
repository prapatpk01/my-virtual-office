from .base import BaseStrategy, Signal, SignalType
from .wt_adx_strategy import WTADXStrategy
from .ai_signal import AISignalStrategy
from .ai_expert_strategy import AIExpertStrategy
from .trend_confirm_strategy import TrendConfirmStrategy
from .sentinel_strategy import SentinelStrategy
from .sentinel_two_target import install_sentinel_two_target

# Sentinel-only lifecycle overlay:
# TP1 = +1.0R trim 60%, lock remaining SL at +0.3R, then keep the native
# mapped R1/S1 target or OPEN_SKY/OPEN_FLOOR dynamic runner as TP2.
install_sentinel_two_target(SentinelStrategy)

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
