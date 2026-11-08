# written by claude
"""
evaluate_qtable.py -- Q-table conversion + deterministic evaluation for Q-SpiRL.

Follows the evaluation protocol of the Q-SpiRL paper (Sec. III-D to III-F):
  1. Convert a trained network into an explicit Q-table over all 8*5*8*8 = 2,560
     discrete states (one offline encoder realization, set by --enc_seed).
  2. Run ONE greedy episode (argmax over the table, no exploration, no action
     masking) on each of 100 held-out environments, seeds 2000..2099.
  3. Report success rate (SR), success-weighted path length (SPL), mean path
     length over successes (PL, metres) and mean turn rate over successes (TR).

Run from the repository root (next to train.py, env.py, models/, helper/).
Reuse the flags you trained with (model/encoder/env flags); training-only flags
such as --episodes or --model_dir are ignored and listed so you can check them.

Examples
--------
  # evaluate a trained checkpoint (20x20, paper settings)
  python evaluate_qtable.py --checkpoint runs/fix_q8_g20_seed0/models/last_....pth \
      --quantum --qubits 8 --num_layers 3 --type mean --max_freq 100 \
      --num_timesteps 10 --dt 0.1 --mem_tau 0.04 --syn_tau 0.01 \
      --num_static 6 --num_dynamic 1 --grid_size 20 --save_qtable

  # re-evaluate a saved (e.g. noisy) Q-table without touching the model
  python evaluate_qtable.py --qtable eval_out/qtable.npy --grid_size 20 \
      --num_static 6 --num_dynamic 1

Assumptions you should know about (see README text in the chat):
  * Shortest feasible path L* (for SPL) ignores dynamic obstacles and heading
    limits: 8-connected Dijkstra on the static obstacles only.
  * By default the training-only diagonal obstacle is NOT placed in test
    environments (paper text). Use --with_diag to include it.
"""
import argparse
import csv
import heapq
import itertools
import json
import math
import os
import sys
from collections import Counter

import numpy as np

STATE_DIMS = (8, 5, 8, 8)   # R_o in 1..8, D_o in 0..4, R_T in 1..8, A_T->o in 1..8
NUM_ACTIONS = 5
FORWARD_ACTION = 2          # zero-yaw action (index into the 5 actions)
# Same constant as helper/misc.py (24.6 m per 28*sqrt(2) grid units).
PATH_SCALE = 24.6 / np.linalg.norm(np.array([28.0, 28.0]))


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="Trained .pth state_dict to convert.")
    src.add_argument("--qtable", type=str, help="Pre-built Q-table .npy, shape (8,5,8,8,5).")

    # model flags (defaults mirror train.py / helper/misc.py -- pass what you trained with)
    p.add_argument("--qmlp", action="store_true")
    p.add_argument("--quantum", action="store_true")
    p.add_argument("--qubits", type=int, default=3)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_actions", type=int, default=5)
    p.add_argument("--mem_tau", type=float, default=0.02)
    p.add_argument("--syn_tau", type=float, default=0.01)
    p.add_argument("--type", type=str, default="max", choices=["mean", "max", "last"])
    p.add_argument("--input_size", type=int, default=29)

    # encoder flags
    p.add_argument("--max_freq", type=float, default=100.0)
    p.add_argument("--num_timesteps", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--bg_freq", type=float, default=0.0)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--enc_seed", type=int, default=42)

    # environment
    p.add_argument("--grid_size", type=int, default=20)
    p.add_argument("--num_static", type=int, default=10)
    p.add_argument("--num_dynamic", type=int, default=1)
    p.add_argument("--with_diag", action="store_true",
                   help="Include the training-only diagonal obstacle at test time.")

    # evaluation
    p.add_argument("--seed_start", type=int, default=2000)
    p.add_argument("--num_envs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out", type=str, default="eval_out")
    p.add_argument("--save_qtable", action="store_true")

    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[info] ignoring flags not used by evaluation: {' '.join(unknown)}")
    return args


# --------------------------------------------------------------------------
# Q-table construction (the only model-dependent part)
# --------------------------------------------------------------------------
def all_states():
    """All 2,560 discrete states in a fixed order, as float32 rows [R_o, D_o, R_T, A]."""
    return np.array(list(itertools.product(range(1, 9), range(0, 5),
                                           range(1, 9), range(1, 9))), dtype=np.float32)


def build_qtable(args):
    import torch
    from helper.encode import freq_encoder, encode_one_hot
    from models.vanilla_models import QMLP, QSNN
    from models.quantum_models import HybridMLP, HybridSNN

    if args.qmlp:
        mod = HybridMLP if args.quantum else QMLP
    else:
        mod = HybridSNN if args.quantum else QSNN

    net_params = {
        "input_size": args.input_size,
        "hidden_size": 30 if not args.quantum else 35,
        "num_layers": args.num_layers,
        "num_actions": args.num_actions,
        "qubits": args.qubits,
    }
    if not args.qmlp:
        net_params["tau_mem"] = args.mem_tau
        net_params["tau_syn"] = args.syn_tau
        net_params["type"] = args.type

    print(f"[model] {mod.__name__} {net_params}")
    net = mod(**net_params)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()

    enc = None
    if not args.qmlp:
        enc = freq_encoder(max_freq=args.max_freq, num_timesteps=args.num_timesteps,
                           dt=args.dt, bg_freq=args.bg_freq, noise_std=args.noise_std,
                           seed=args.enc_seed)

    states = all_states()
    feats = []
    for s in states:
        f = enc.encode_state(s) if enc is not None else encode_one_hot(s)
        feats.append(np.asarray(f, dtype=np.float32))
    X = torch.from_numpy(np.stack(feats))
    print(f"[qtable] encoded features: {tuple(X.shape)} (states, ...)")

    outs = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            q = net(X[i:i + args.batch_size])
            outs.append(q.detach().cpu().numpy().reshape(-1, NUM_ACTIONS))
    Q = np.concatenate(outs).reshape(STATE_DIMS + (NUM_ACTIONS,))
    return Q


# --------------------------------------------------------------------------
# Evaluation (no torch needed)
# --------------------------------------------------------------------------
def shortest_path_len(static_obstacles, start, goal, W, H):
    """8-connected Dijkstra on static obstacles only (grid units).
    Axis moves cost 1, diagonal moves sqrt(2); corner cutting allowed because
    the environment only checks the destination cell."""
    blocked = set(map(tuple, np.asarray(static_obstacles, dtype=int).reshape(-1, 2).tolist()))
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if start in blocked or goal in blocked:
        return math.inf
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if (x, y) == goal:
            return d
        if d > dist.get((x, y), math.inf):
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H) or (nx, ny) in blocked:
                    continue
                nd = d + (1.0 if dx == 0 or dy == 0 else math.sqrt(2.0))
                if nd < dist.get((nx, ny), math.inf):
                    dist[(nx, ny)] = nd
                    heapq.heappush(heap, (nd, (nx, ny)))
    return math.inf


def path_length_grid(positions):
    pos = np.asarray(positions, dtype=float)
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(pos[1:] - pos[:-1], axis=1).sum())


def make_env(args):
    os.environ["GRID_SIZE"] = str(args.grid_size)   # read by the edited env.py
    from env import GridEnvCells
    env = GridEnvCells(num_static=args.num_static, num_dynamic=args.num_dynamic,
                       seed=args.seed_start)
    if env.W != args.grid_size:
        sys.exit(f"[error] env.py built a {env.W}x{env.H} grid but --grid_size is "
                 f"{args.grid_size}. Apply the GRID_SIZE edit to env.py first.")
    if tuple(env.terminal_cell) != (env.W - 1, env.H - 1):
        sys.exit(f"[error] goal cell is {tuple(env.terminal_cell)}, expected "
                 f"{(env.W - 1, env.H - 1)}. Fix terminal_cell in env.py.")
    if hasattr(env, "use_diag"):
        env.use_diag = bool(args.with_diag)
    elif not args.with_diag:
        print("[warn] env.py has no 'use_diag' flag, so the training-only diagonal "
              "obstacle WILL appear in test environments. Add the use_diag edit "
              "or pass --with_diag to acknowledge this.")
    env.max_steps = args.max_steps
    return env


def greedy_episode(env, Q, seed, max_steps):
    state = env.reset(seed=seed)
    lstar = shortest_path_len(env.static_obstacles, env.start_cell, env.terminal_cell,
                              env.W, env.H)
    positions = [env.pos.copy()]
    actions = []
    done, info, steps = False, {}, 0
    while not done and steps < max_steps:
        a = int(np.argmax(Q[state[0] - 1, state[1], state[2] - 1, state[3] - 1]))
        state, _, done, info = env.step(a)
        actions.append(a)
        positions.append(env.pos.copy())
        steps += 1

    if info.get("goal"):
        outcome = "goal"
    elif info.get("out_of_bounds"):
        outcome = "out_of_bounds"
    elif info.get("collision"):
        outcome = "static" if info.get("collision_type") == "static" else "dynamic"
    else:
        outcome = "timeout"

    success = outcome == "goal"
    L = path_length_grid(positions)
    spl = (lstar / max(L, lstar)) if (success and math.isfinite(lstar)) else 0.0
    turn_rate = float(np.mean([a != FORWARD_ACTION for a in actions])) if actions else 0.0
    return {
        "seed": seed, "outcome": outcome, "success": int(success), "steps": steps,
        "path_grid": L, "path_m": L * PATH_SCALE, "lstar_grid": lstar,
        "spl": spl, "turn_rate": turn_rate,
    }


def evaluate(Q, args):
    env = make_env(args)
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.num_envs):
        rows.append(greedy_episode(env, Q, seed, args.max_steps))
    return rows


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def summarize(rows):
    valid = [r for r in rows if math.isfinite(r["lstar_grid"])]
    infeasible = len(rows) - len(valid)
    n = len(valid)
    succ = [r for r in valid if r["success"]]
    sr = len(succ) / n if n else float("nan")
    sr_se = math.sqrt(sr * (1 - sr) / n) if n else float("nan")
    spl, spl_se = _mean_se([r["spl"] for r in valid])
    pl, pl_se = _mean_se([r["path_m"] for r in succ])
    tr, tr_se = _mean_se([r["turn_rate"] for r in succ])
    outcomes = Counter(r["outcome"] for r in valid)
    return {
        "n_envs": n, "n_infeasible_excluded": infeasible,
        "SR": sr, "SR_se": sr_se, "SPL": spl, "SPL_se": spl_se,
        "PL_m": pl, "PL_m_se": pl_se, "TR": tr, "TR_se": tr_se,
        "outcomes": {k: outcomes.get(k, 0) for k in
                     ["goal", "static", "dynamic", "out_of_bounds", "timeout"]},
    }


def print_summary(s, args):
    o = s["outcomes"]
    print("\n" + "=" * 72)
    print(f" Greedy Q-table evaluation: {s['n_envs']} held-out envs, seeds "
          f"{args.seed_start}..{args.seed_start + args.num_envs - 1}, "
          f"{args.grid_size}x{args.grid_size}, ns={args.num_static}, nd={args.num_dynamic}, "
          f"diag={'yes' if args.with_diag else 'no'}")
    print("=" * 72)
    print(f"  Success rate (SR) : {s['SR']:.3f} +/- {s['SR_se']:.3f}")
    print(f"  SPL               : {s['SPL']:.4f} +/- {s['SPL_se']:.4f}")
    print(f"  Path length (PL)  : {s['PL_m']:.2f} m +/- {s['PL_m_se']:.2f}  (successes only)")
    print(f"  Turn rate (TR)    : {s['TR']:.4f} +/- {s['TR_se']:.4f}  (successes only)")
    print(f"  Outcomes          : goal {o['goal']} | static {o['static']} | dynamic "
          f"{o['dynamic']} | out_of_bounds {o['out_of_bounds']} | timeout {o['timeout']}")
    if s["n_infeasible_excluded"]:
        print(f"  Excluded (no feasible path): {s['n_infeasible_excluded']}")
    print("=" * 72)


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.qtable:
        Q = np.load(args.qtable)
        if Q.shape != STATE_DIMS + (NUM_ACTIONS,):
            sys.exit(f"[error] Q-table shape {Q.shape}, expected {STATE_DIMS + (NUM_ACTIONS,)}")
    else:
        Q = build_qtable(args)
        if args.save_qtable:
            np.save(os.path.join(args.out, "qtable.npy"), Q)
            print(f"[saved] {os.path.join(args.out, 'qtable.npy')}")

    rows = evaluate(Q, args)
    summary = summarize(rows)
    print_summary(summary, args)

    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2, default=float)
    print(f"[saved] {args.out}/episodes.csv and summary.json")
    return summary


if __name__ == "__main__":
    main()