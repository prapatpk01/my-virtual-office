"""Unified launcher — pick the trading system with the MODE env var.

    MODE=dual    -> DUAL ENTRY PRECISION V1.4   (default)
    MODE=regime  -> Signal Regime Bias bot (legacy)

This file is baked into the Docker image AS /app/main.py, so it runs no
matter which start command Railway uses (`python main.py`, the image CMD,
or railway.json's startCommand) — switching systems is ONLY an env-var
change + redeploy, never a Dockerfile/start-command hunt.

Pure stdlib; replaces itself with the chosen bot via os.execv so signals
and restart policies behave exactly as if the bot was started directly.
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
        if not os.path.isdir(os.path.join(HERE, "dual_entry_v14")):
            print(f"FATAL: dual_entry_v14/ not found under {HERE}", flush=True)
            sys.exit(1)
    elif mode in ("regime", "regime_bias", "legacy", "old"):
        target_dir = os.path.join(HERE, "signal_regime_bot")
        argv = [sys.executable, "main.py"]
        name = "Signal Regime Bias (legacy)"
        if not os.path.isfile(os.path.join(target_dir, "main.py")):
            print(f"FATAL: signal_regime_bot/main.py not found under {HERE}", flush=True)
            sys.exit(1)
    else:
        print(f"FATAL: unknown MODE={mode!r} — use MODE=dual or MODE=regime", flush=True)
        sys.exit(1)

    print(f"{banner}\nLAUNCHER: MODE={mode} -> {name}\n"
          f"  cwd:  {target_dir}\n  exec: {' '.join(argv)}\n{banner}", flush=True)

    if os.environ.get("LAUNCHER_DRY_RUN") == "1":
        print("LAUNCHER_DRY_RUN=1 — not executing.", flush=True)
        return

    os.chdir(target_dir)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
