"""Replay tournament simulations with Group 4's voluntary discards switched off.

Every finite-budget simulation in which Group 4 holds most seats (the 2-, 9-, 18-
and 36-roommate all-Group-4 households, and the nine-roommate households with
eight Group 4 players) is replayed with the same roster, seat order, drawer,
budget, duration and seed. Group 4 seats run the submitted player with
`_discard_policy` returning a zero allowance; holes are still replaced. The
nine-roommate households with a single Group 4 player are replayed too.

The simulator and the other groups' players must be the tournament versions, so
run this from a checkout of the course repository's final main branch:

    python /path/to/report/discard_ablation.py results/7_raw_runs.csv out.csv --workers 8

`--check N` also replays N randomly chosen simulations with the unmodified
player and fails unless every seat reproduces the tournament score.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402

PLAYERS, _ = discover()
SUBMITTED = PLAYERS["4"]


class NoVoluntaryDiscards(SUBMITTED):
    def _discard_policy(self, turn):
        return 0, 0.0


HOUSEHOLDS = {"n2_homo", "n9_homo", "n18_homo", "n36_homo", "n9_minority"}


def replay(job):
    variant, sim_id, groups, unit, capacity, budget, days, seed = job
    group4 = NoVoluntaryDiscards if variant == "no_voluntary_discards" else SUBMITTED
    players = [group4 if g == "4" else PLAYERS[g] for g in groups]
    result = Engine(players=players, capacity=capacity, selection_unit=unit, days=days,
                    seed=seed, timeout=5.0, budget=budget, keep_records=False).run()
    seats = result["players"]
    return variant, sim_id, groups, seats, result["total_spent"], result["budget_exhausted_on_day"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_runs")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", type=int, default=40)
    args = parser.parse_args()

    seats, meta, scores = defaultdict(dict), {}, defaultdict(dict)
    with open(args.raw_runs) as f:
        for row in csv.DictReader(f):
            if row["household_type"] not in HOUSEHOLDS or row["budget_total"] == "inf":
                continue
            if "4" not in row["roster"].split("+"):
                continue
            sim = row["sim_id"]
            seats[sim][int(row["seat"])] = row["group"]
            scores[sim][int(row["seat"])] = float(row["embarrassment_per_day"])
            meta[sim] = (int(row["unit"]), int(row["drawer_socks"]), float(row["budget_total"]),
                         int(row["days"]), int(row["seed"]))

    sims = sorted(seats, key=int)
    random.seed(0)
    checked = set(random.sample(sims, min(args.check, len(sims))))
    jobs = []
    for sim in sims:
        groups = [seats[sim][k] for k in sorted(seats[sim])]
        jobs.append(("no_voluntary_discards", sim, groups) + meta[sim])
        if sim in checked:
            jobs.append(("submitted", sim, groups) + meta[sim])
    jobs.sort(key=lambda j: -len(j[2]) * j[6])

    with Pool(args.workers) as pool, open(args.output, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["sim_id", "variant", "group4_score", "group4_sockless_days",
                      "others_score", "household_total_spent", "budget_exhausted_on_day"])
        for variant, sim, groups, result, spent, exhausted in pool.imap_unordered(replay, jobs):
            mine = [s for g, s in zip(groups, result) if g == "4"]
            others = [s for g, s in zip(groups, result) if g != "4"]
            if variant == "submitted":
                expected = [scores[sim][k] for k in sorted(scores[sim])]
                got = [s["mean_daily_embarrassment"] for s in result]
                assert all(abs(a - b) < 1e-4 for a, b in zip(got, expected)), sim
            out.writerow([sim, variant,
                          sum(s["mean_daily_embarrassment"] for s in mine) / len(mine),
                          sum(s["sockless_days"] for s in mine) / len(mine),
                          (sum(s["mean_daily_embarrassment"] for s in others) / len(others)) if others else "",
                          spent, exhausted if exhausted is not None else ""])


if __name__ == "__main__":
    main()
