"""Production entry point for WT Trend Entry + Trend Confirm.

All strategy creation, quotas, hedge rules, sleep mode and configuration are
installed by run_dual_bot. This filename is retained so the existing Railway
start command does not need to change.
"""
from __future__ import annotations

import asyncio

import run_dual_bot  # installs patches on run_bot
import run_bot


if __name__ == "__main__":
    try:
        asyncio.run(run_bot.main())
    except KeyboardInterrupt:
        pass
