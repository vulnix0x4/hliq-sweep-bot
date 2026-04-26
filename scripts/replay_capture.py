#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hliq_bot.bot import SweepBot
from hliq_bot.config import load_config
from hliq_bot.replay import load_market_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay captured Hyperliquid market events through the bot.")
    parser.add_argument("--input", default=None, help="JSONL market capture path. Defaults to BOT_REPLAY_INPUT_PATH.")
    parser.add_argument(
        "--journal",
        default=None,
        help="Optional journal output path for replay results. Defaults to runtime/replay_signals.jsonl.",
    )
    parser.add_argument("--append", action="store_true", help="Append to the replay journal instead of replacing it.")
    args = parser.parse_args()

    cfg = load_config()
    cfg.mode = "replay"
    if args.input:
        cfg.replay.input_path = args.input
    if args.journal:
        cfg.runtime.journal_path = args.journal
    else:
        cfg.runtime.journal_path = str(Path(cfg.runtime.runtime_dir) / "replay_signals.jsonl")
    cfg.runtime.market_capture_enabled = False
    out_path = Path(cfg.runtime.journal_path)
    if not args.append and out_path.exists():
        out_path.unlink()

    bot = SweepBot(cfg)
    summary = bot.run_replay(load_market_events(cfg.replay.input_path))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
