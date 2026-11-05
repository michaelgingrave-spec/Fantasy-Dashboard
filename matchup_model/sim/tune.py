"""Fit the simulator's noise parameters to past games: a parallel coordinate search that lowers `calibrate.score` (the gap between the simulated and the real correlations of
players' stats, the spread of outcomes around the simulated distribution, and the team-level volume bias).

    python -m matchup_model.sim.tune 2023 2024        # about 2 minutes a round on 8 cores; writes data/dfs/.cache/sim/tuned_params.json after every round

Only the seasons named are used, so the others stay a clean test (`validate.py`). The shipped noise parameters in `engine.SimParams` are the result of this search on 2023-24
(five rounds, loss 2.01 -> 1.27); copy the printed values there to adopt a new fit.

`vol_pass` / `vol_rush` are not searched: they are 1.0 (every player's mean is his projection; see the engine's module note), which makes the simulated TEAM volumes run 6-13% above
history and adds a constant to the loss's team-volume term (0.12 -> 2.76 on 2023-24). The other noise parameters were not re-tuned after that change; their z variances improved
(0.149 -> 0.108), so compare a re-tune with the loss parts, not the total.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from matchup_model.sim import calibrate as C
from matchup_model.sim import engine as E
from matchup_model.sim import history as H

SEASONS: list[int] = [2023, 2024]
EVERY = 2                                  # every second game is enough, and quicker
RECS = None

# name: (kind, step, lo, hi): a multiplicative or additive step and the allowed range
SPACE = {"pass_sd": ("mul", 1.3, 0.04, 0.35), "rush_sd": ("mul", 1.3, 0.05, 0.40), "pass_G": ("add", 0.025, 0.0, 0.2), "rush_G": ("add", 0.025, 0.0, 0.2),
         "pass_script": ("mul", 1.35, 0.02, 0.4), "rush_script": ("mul", 1.35, 0.03, 0.5), "kappa_tgt": ("mul", 1.5, 8, 400), "kappa_car": ("mul", 1.5, 8, 400),
         "cr_eff": ("mul", 1.4, 0.02, 0.5), "ypc_eff": ("mul", 1.4, 0.02, 0.5), "ypc_sd": ("mul", 1.3, 0.05, 0.5), "rec_shape": ("mul", 1.3, 0.8, 6),
         "ypc_r_sd": ("mul", 1.3, 0.05, 0.6), "rush_shape": ("mul", 1.3, 0.6, 6), "ypc_r_eff": ("mul", 1.5, 0.0, 0.4), "rush_rho": ("add", 0.15, -0.5, 0.8),
         "a_yards": ("add", 0.08, 0.1, 0.8), "a_script": ("add", 0.08, 0.0, 0.6), "sd_td": ("mul", 1.08, 0.9, 2.0), "td_pass_link": ("mul", 1.4, 0.05, 3.0)}


def _init(seasons: list[int]) -> None:
    global RECS
    RECS = H.load(seasons)[::EVERY]


def _evaluate(kv: dict) -> float:
    return C.score(C.diagnose(RECS, params=E.SimParams().with_(**kv), n=1500, seed=5))


def neighbours(cur: dict) -> list[dict]:
    out = []
    for k, (kind, step, lo, hi) in SPACE.items():
        for sgn in (+1, -1):
            v = cur[k] * (step if sgn > 0 else 1 / step) if kind == "mul" else cur[k] + sgn * step
            v = min(max(v, lo), hi)
            if abs(v - cur[k]) > 1e-9:
                out.append({**cur, k: v})
    return out


def tune(seasons: list[int], rounds: int = 5, workers: int = 7, start: dict | None = None, out_path=None) -> dict:
    cur = start or {k: getattr(E.SimParams(), k) for k in SPACE}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(seasons,)) as ex:
        best = list(ex.map(_evaluate, [cur]))[0]
        print(f"start loss {best:.3f}", flush=True)
        for rnd in range(1, rounds + 1):
            cands = neighbours(cur)
            losses = list(ex.map(_evaluate, cands))
            improved = [(losses[i], cands[i]) for i in np.argsort(losses) if losses[i] < best - 0.01]
            if not improved:
                print(f"round {rnd}: no improvement; stop", flush=True)
                break
            moves: dict = {}                                    # try the best three non-conflicting moves together
            for _, c in improved:
                k = next(kk for kk in SPACE if abs(c[kk] - cur[kk]) > 1e-9)
                if k not in moves and len(moves) < 3:
                    moves[k] = c[k]
            combo = {**cur, **moves}
            lc = list(ex.map(_evaluate, [combo]))[0] if len(moves) > 1 else improved[0][0]
            cur, best = (combo, lc) if (len(moves) > 1 and lc < improved[0][0]) else (improved[0][1], improved[0][0])
            print(f"round {rnd}: loss {best:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    return cur


if __name__ == "__main__":
    from dfs.config import ROOT
    seasons = [int(x) for x in sys.argv[1:]] or SEASONS
    final = tune(seasons, out_path=ROOT / "data" / "dfs" / ".cache" / "sim" / "tuned_params.json")
    print(json.dumps(final, indent=1))
