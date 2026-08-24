"""Production trading launcher.

EMA Hybrid Pro is the production runtime. Legacy runtimes are available only
when ALLOW_NON_EMA_RUNTIME=1 is explicitly set, which prevents Railway from
silently running an older strategy when MODE is stale or misconfigured.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _legacy_allowed() -> bool:
    return os.environ.get("ALLOW_NON_EMA_RUNTIME", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def main() -> None:
    requested = os.environ.get("MODE", "ema_hybrid").strip().lower()
    # Production safety: an old Railway MODE value must not silently launch
    # Adaptive/TPC/Dual/legacy code. Opt out explicitly only for development.
    if requested not in ("ema_hybrid", "ema_hybrid_pro", "hybrid") and not _legacy_allowed():
        print(
            f"[BOOT] MODE={requested!r} is not allowed in production. "
            "Forcing MODE=ema_hybrid. Set ALLOW_NON_EMA_RUNTIME=1 only for "
            "intentional legacy/development runs.",
            flush=True,
        )
        requested = "ema_hybrid"

    mode = requested
    banner = "=" * 70
    if mode in ("dual", "dual_entry", "v14", "v1.4"):
        target_dir = HERE
        argv = [sys.executable, "-m", "dual_entry_v14.main"]
        name = "DUAL ENTRY PRECISION V1.4"
    elif mode in ("regime", "regime_bias", "legacy", "old"):
        target_dir = os.path.join(HERE, "signal_regime_bot")
        argv = [sys.executable, "main.py"]
        name = "Signal Regime Bias"
    elif mode in ("htf", "htf_pullback", "simple"):
        target_dir = os.path.join(HERE, "htf_bot")
        argv = [sys.executable, "main.py"]
        name = "HTF Pullback"
    elif mode in ("tpc", "trend_pullback_continuation"):
        target_dir = os.path.join(HERE, "hma_bot")
        argv = [sys.executable, "main_v16.py"]
        name = "Trend Pullback Continuation (TPC Sentinel)"
    elif mode in ("ema_hybrid", "ema_hybrid_pro", "hybrid"):
        target_dir = os.path.join(HERE, "ema_hybrid_bot")
        argv = [sys.executable, "main_guarded.py"]
        name = "EMA Hybrid Pro — One-Position Guarded"
    elif mode in ("hma", "hma16", "trendfollow"):
        target_dir = os.path.join(HERE, "hma_bot")
        argv = [sys.executable, "main_v16.py"]
        name = "Trend Pullback Continuation (legacy HMA alias)"
    else:
        print(f"FATAL: unknown MODE={mode!r}", flush=True)
        sys.exit(1)

    target = os.path.join(target_dir, argv[-1]) if argv[-1].endswith(".py") else target_dir
    if argv[-1].endswith(".py") and not os.path.isfile(target):
        print(f"FATAL: target not found: {target}", flush=True)
        sys.exit(1)

    print(
        f"{banner}\n[BOOT] RUNTIME_ID=EMA_HYBRID_PRO\n"
        f"[BOOT] REQUESTED_MODE={os.environ.get('MODE', '<unset>')}\n"
        f"[BOOT] EFFECTIVE_MODE={mode}\n"
        f"[BOOT] STRATEGY={name}\n"
        f"[BOOT] cwd={target_dir}\n[BOOT] exec={' '.join(argv)}\n{banner}",
        flush=True,
    )
    os.chdir(target_dir)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
