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


_install_detailed_stats_renderer()
