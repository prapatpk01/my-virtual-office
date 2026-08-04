"""Unified launcher — pick the trading system with the MODE env var.

    MODE=dual    -> DUAL ENTRY PRECISION V1.4   (default)
    MODE=regime  -> Signal Regime Bias bot (legacy)
    MODE=htf     -> HTF pullback bot (1H/4H, backtest-validated)
    MODE=hma     -> HMA Gate Sentinel

This file is baked into the Docker image AS /app/main.py, so it runs no
matter which start command Railway uses. Switching systems is only an env-var
change plus redeploy.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    mode = os.environ.get("MODE", "dual").strip().lower()
    banner = "=" * 62
    if mode in ("dual", "dual_entry", "v14", "v1.4"):
        target_dir = HERE
        argv = [sys.executable, "-m", "dual_entry_v14.main"]
        name = "DUAL ENTRY PRECISION V1.4"
        required = os.path.join(HERE, "dual_entry_v14")
    elif mode in ("regime", "regime_bias", "legacy", "old"):
        target_dir = os.path.join(HERE, "signal_regime_bot")
        argv = [sys.executable, "main.py"]
        name = "Signal Regime Bias (legacy)"
        required = os.path.join(target_dir, "main.py")
    elif mode in ("htf", "htf_pullback", "simple"):
        target_dir = os.path.join(HERE, "htf_bot")
        argv = [sys.executable, "main.py"]
        name = "HTF Pullback (1H/4H, backtest-validated)"
        required = os.path.join(target_dir, "main.py")
    elif mode in ("hma", "hma16", "trendfollow"):
        target_dir = os.path.join(HERE, "hma_bot")
        argv = [sys.executable, "main.py"]
        name = "HMA Gate Sentinel"
        required = os.path.join(target_dir, "main.py")
    else:
        print(f"FATAL: unknown MODE={mode!r} — use MODE=dual, regime, htf or hma", flush=True)
        sys.exit(1)

    if not os.path.exists(required):
        print(f"FATAL: required target not found: {required}", flush=True)
        sys.exit(1)

    print(
        f"{banner}\nLAUNCHER: MODE={mode} -> {name}\n"
        f"  cwd:  {target_dir}\n  exec: {' '.join(argv)}\n{banner}",
        flush=True,
    )

    if os.environ.get("LAUNCHER_DRY_RUN") == "1":
        print("LAUNCHER_DRY_RUN=1 — not executing.", flush=True)
        return

    os.chdir(target_dir)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
