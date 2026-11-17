"""Offline benchmark (no LLM): Random / FrequencyBR / BSI(seed library, = B2) / Oracle across bot families.

    .venv/bin/python scripts/demo_offline.py --game rps3 --split dev --seeds 3 --horizon 150
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from bsi.sandbox import SandboxPool
from bsi.store import TraceWriter, make_run_dir
from p2_opponent.agents import FrequencyBRAgent, OracleBRAgent, PosteriorBRAgent, RandomAgent
from p2_opponent.bots import FAMILIES, sample_bot
from p2_opponent.games import make_game
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import SEED_LIBRARY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="rps3")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    game = make_game(args.game)
    run = make_run_dir("runs", "p2_offline", vars(args))
    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    t0 = time.time()
    with SandboxPool(n_workers=2) as pool, TraceWriter(run / "episodes.jsonl.gz") as w:
        for family in args.families.split(","):
            for s in range(args.seeds):
                bot = sample_bot(game, family, args.split, np.random.default_rng(1000 + s), args.horizon)
                agents = [RandomAgent(), FrequencyBRAgent(), PosteriorBRAgent(pool, SEED_LIBRARY, name="bsi_seedlib"), OracleBRAgent(bot)]
                for agent in agents:
                    res = run_episode(game, bot, agent, args.horizon, seed=s, writer=w)
                    agent.close()
                    rows[(family, agent.name)].append(res.summary())
    agent_names = ["random", "freq_br", "bsi_seedlib", "oracle_br"]
    print(f"\n{args.game} / {args.split} / {args.seeds} seeds x {args.horizon} rounds   ({time.time() - t0:.0f}s)   traces: {run}")
    print(f"{'family':<12}" + "".join(f"{a:>26}" for a in agent_names))
    print(f"{'':<12}" + "".join(f"{'payoff  regret  KL(2nd½)':>26}" for _ in agent_names))
    for family in args.families.split(","):
        cells = []
        for a in agent_names:
            r = rows[(family, a)]
            pay = np.mean([x["payoff_per_round"] for x in r])
            reg = np.mean([x["regret"] for x in r])
            k = np.nanmean([x["kl_second_half"] for x in r]) if a != "random" else float("nan")
            cells.append(f"{pay:>8.3f}{reg:>8.3f}{k:>10.3f}")
        print(f"{family:<12}" + "".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
