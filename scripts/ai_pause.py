#!/usr/bin/env python3
"""Operator pause/resume for the AI strategy.

Writes runtime/ai_override.flag with one of:
  - "pause"          — no new opens; existing positions managed normally
  - "no_new"         — same as pause; alias
  - "close_all"      — bot closes all positions on next AI tick, then stops
                       opening new ones until cleared
  - (file removed)   — normal trading resumes

USAGE
-----
    python scripts/ai_pause.py pause      # block new opens
    python scripts/ai_pause.py close-all  # force-close everything
    python scripts/ai_pause.py resume     # clear flag
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

VALID = {"pause", "no-new", "close-all", "resume"}


def main() -> int:
    parser = argparse.ArgumentParser(description="AI override flag.")
    parser.add_argument("action", choices=sorted(VALID))
    parser.add_argument("--runtime-dir", default="runtime")
    args = parser.parse_args()

    flag = ROOT / args.runtime_dir / "ai_override.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "resume":
        if flag.exists():
            flag.unlink()
            print(f"Cleared {flag}. AI resumes normal trading on next tick.")
        else:
            print(f"No override flag present at {flag}. AI is already running normally.")
        return 0

    # Normalize names
    value = args.action.replace("-", "_")
    flag.write_text(value + "\n", encoding="utf-8")
    print(f"Wrote '{value}' to {flag}.")
    if value == "pause" or value == "no_new":
        print("  AI will skip OPENING new positions but continue managing existing ones.")
    elif value == "close_all":
        print("  AI will FORCE-CLOSE all positions on next decision tick (one per coin per interval_sec).")
        print("  When all are flat, AI will stop opening new positions until you `resume`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
