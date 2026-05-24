#!/usr/bin/env python3
"""Emergency flatten: instruct the AI strategy to close every open position
on the next decision tick, then refuse to open new ones until cleared.

This writes runtime/ai_override.flag with 'close_all'. The bot picks it up
within its normal AI poll cycle (default 5 min per coin). For an instant
flatten outside the bot's control, use HL's web UI.

USAGE
-----
    python scripts/ai_flatten.py
    python scripts/ai_flatten.py --runtime-dir /var/lib/hl/runtime
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Emergency flatten — sets AI override to close_all.")
    parser.add_argument("--runtime-dir", default="runtime")
    args = parser.parse_args()

    # Delegate to ai_pause.py to keep flag semantics in one place.
    rc = subprocess.call([
        sys.executable, str(ROOT / "scripts" / "ai_pause.py"),
        "close-all", "--runtime-dir", args.runtime_dir,
    ])
    if rc != 0:
        return rc
    print()
    print("To resume trading later: python scripts/ai_pause.py resume")
    print("To see status:           python scripts/ai_status.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
