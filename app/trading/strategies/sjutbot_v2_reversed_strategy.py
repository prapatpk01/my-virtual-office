"""
SJUTBot v2 Reversed — Contrarian wrapper.

Every signal from v2 is flipped:
  BUY  → SELL (open short)
  SELL → BUY  (open long)

SL/TP are reflected symmetrically through current price:
  new_sl = 2×price − old_sl
  new_tp = 2×price − old_tp

Useful for backtesting whether fading the UT-Bot signal beats following it.
"""
from .base import Signal, SignalType
from .sjutbot_strategy import SJUTBotStrategy


class SJUTBotV2ReversedStrategy(SJUTBotStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.name = (params or {}).get("name", "SJUTBotV2ReversedStrategy")

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        sig = await super().analyze(candles, current_price, mtf_candles)

        if sig.type == SignalType.HOLD:
            return sig

        # Reflect SL/TP through current price so the reversed side is valid
        meta = dict(sig.metadata or {})
        sl = meta.get("stop_loss")
        tp = meta.get("take_profit")
        if sl is not None and tp is not None:
            meta["stop_loss"]   = round(2 * current_price - sl, 2)
            meta["take_profit"] = round(2 * current_price - tp, 2)

        new_type = SignalType.SELL if sig.type == SignalType.BUY else SignalType.BUY
        return Signal(
            new_type, sig.symbol, sig.price,
            amount=sig.amount,
            reason=f"[REV] {sig.reason}",
            confidence=sig.confidence,
            metadata=meta,
        )
