#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hliq_bot.bot import SweepBot
from hliq_bot.config import load_config
from hliq_bot.replay import load_market_events


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay captured Hyperliquid market events through the bot.")
    parser.add_argument("--input", default=None, help="JSONL market capture path. Defaults to BOT_REPLAY_INPUT_PATH.")
    parser.add_argument(
        "--journal",
        default=None,
        help="Optional journal output path for replay results. Defaults to runtime/replay_signals.jsonl.",
    )
    parser.add_argument("--append", action="store_true", help="Append to the replay journal instead of replacing it.")
    parser.add_argument(
        "--allow-untagged-events",
        action="store_true",
        help="Replay legacy events without a coin field. Unsafe for multi-coin captures.",
    )
    parser.add_argument(
        "--ignore-runtime-pause",
        action="store_true",
        help="Use an isolated pause file so replay evaluates strategy policy instead of the live watcher pause.",
    )
    args = parser.parse_args()

    _load_env(ROOT / ".env")
    cfg = load_config()
    cfg.mode = "replay"
    if args.input:
        cfg.replay.input_path = args.input
    if args.journal:
        cfg.runtime.journal_path = args.journal
    else:
        cfg.runtime.journal_path = str(Path(cfg.runtime.runtime_dir) / "replay_signals.jsonl")
    if args.ignore_runtime_pause:
        cfg.runtime.trade_pause_path = str(Path(cfg.runtime.runtime_dir) / "replay_pause_ignored.flag")
    cfg.runtime.market_capture_enabled = False
    out_path = Path(cfg.runtime.journal_path)
    if not args.append and out_path.exists():
        out_path.unlink()

    bot = SweepBot(cfg)
    require_coin = len(cfg.feed.coins) > 1 and not args.allow_untagged_events
    summary = bot.run_replay(load_market_events(cfg.replay.input_path, require_coin=require_coin))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
