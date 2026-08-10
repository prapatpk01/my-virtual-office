# Trading bot package


def _install_detailed_stats_renderer():
    """Patch Telegram's internal/paper /stats renderer at package import time.

    LIVE OKX stats still use adaptive_stats.render_adaptive_stats(); this only
    expands the fallback view used in PAPER mode or before OKX history exists.
    """
    try:
        from .telegram_notifier import TelegramNotifier
        from .stats_renderer import render_internal_stats

        def _render_stats(self, stats: dict) -> str:
            return render_internal_stats(self, stats)

        TelegramNotifier._render_stats = _render_stats
    except Exception:
        # Never make the trading package fail to import because a presentation
        # helper is unavailable. TelegramNotifier keeps its built-in fallback.
        pass


def _install_strategy_label_fix():
    """Make Telegram order captions use the strategy that actually opened it.

    The chart title already receives TradingBot's authoritative position
    strategy key (for example ``Sentinel(XAG/USDT:USDT):L``).  Some Sentinel
    signals also carry an older ``selected_strategy=Trend Confirm`` metadata
    field, and TelegramNotifier previously allowed that metadata to override
    the real strategy in the footer.  Prefer the actual strategy key for
    standalone Sentinel / TrendConfirm orders while preserving selected_strategy
    for routed AI strategies.
    """
    try:
        from .telegram_notifier import TelegramNotifier

        if getattr(TelegramNotifier, "_strategy_label_fix_installed", False):
            return

        original_build_order_caption = TelegramNotifier.build_order_caption

        def _build_order_caption_strategy_safe(self, *args, **kwargs):
            # build_order_caption(self, symbol, side, amount, price, strategy, ...)
            raw_strategy = kwargs.get("strategy")
            if raw_strategy is None and len(args) >= 5:
                raw_strategy = args[4]
            raw = str(raw_strategy or "")

            if raw.startswith("Sentinel(") or raw == "Sentinel":
                kwargs["selected_strategy"] = "Sentinel"
            elif raw.startswith("TrendConfirm(") or raw in ("TrendConfirm", "Trend Confirm"):
                kwargs["selected_strategy"] = "Trend Confirm"

            caption = original_build_order_caption(self, *args, **kwargs)

            # The built-in notifier's exit text is Trend-Confirm-specific.
            # Do not show that policy on Sentinel orders.
            if raw.startswith("Sentinel(") or raw == "Sentinel":
                caption = caption.replace(
                    "🏁 Exit : trend flip (EMA cross-back / close past EMA)",
                    "🏁 Exit : Sentinel TP1 +1.0R trim 60% → SL +0.30R; TP2 = S/R or runner",
                )
            return caption

        TelegramNotifier.build_order_caption = _build_order_caption_strategy_safe
        TelegramNotifier._strategy_label_fix_installed = True
    except Exception:
        # Presentation patches must never prevent the bot from starting.
        pass


_install_detailed_stats_renderer()
_install_strategy_label_fix()
