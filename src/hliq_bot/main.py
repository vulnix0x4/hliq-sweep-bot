from __future__ import annotations

import asyncio
import logging

from hliq_bot.bot import SweepBot
from hliq_bot.config import load_config
from hliq_bot.replay import load_market_events


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


async def _run() -> None:
    cfg = load_config()
    bot = SweepBot(cfg)
    if cfg.mode in {"paper", "live"}:
        await bot.run()
        return
    if cfg.mode == "replay":
        bot.run_replay(load_market_events(cfg.replay.input_path))
        return
    raise ValueError(
        f"Unsupported BOT_MODE={cfg.mode!r}. Supported modes: 'paper', 'live', 'replay'."
    )


def main() -> None:
    _setup_logging()
    asyncio.run(_run())


if __name__ == "__main__":
    main()
