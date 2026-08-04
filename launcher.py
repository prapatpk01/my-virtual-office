"""Unified launcher — pick the trading system with the MODE env var.

MODE=hma launches the HMA Simple Sentinel production entrypoint.
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
    elif mode in ("regime", "regime_bias", "legacy", "old"):
        target_dir = os.path.join(HERE, "signal_regime_bot")
        argv = [sys.executable, "main.py"]
        name = "Signal Regime Bias"
    elif mode in ("htf", "htf_pullback", "simple"):
        target_dir = os.path.join(HERE, "htf_bot")
        argv = [sys.executable, "main.py"]
        name = "HTF Pullback"
    elif mode in ("hma", "hma16", "trendfollow"):
        target_dir = os.path.join(HERE, "hma_bot")
        argv = [sys.executable, "main_v16.py"]
        name = "HMA Simple Sentinel"
    else:
        print(f"FATAL: unknown MODE={mode!r}", flush=True)
        sys.exit(1)

    target = (
        os.path.join(target_dir, argv[-1])
        if argv[-1].endswith(".py")
        else target_dir
    )
    if argv[-1].endswith(".py") and not os.path.isfile(target):
        print(f"FATAL: target not found: {target}", flush=True)
        sys.exit(1)

    print(
        f"{banner}\nLAUNCHER: MODE={mode} -> {name}\n"
        f"  cwd: {target_dir}\n  exec: {' '.join(argv)}\n{banner}",
        flush=True,
    )
    os.chdir(target_dir)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
