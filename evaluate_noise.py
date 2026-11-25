# written by claude
"""
evaluate_noise.py -- evaluate a trained Q-SpiRL checkpoint under simulated quantum noise.

How it works
  1. The checkpoint is loaded and a CLEAN Q-table is built with the original PennyLane
     layer (this equals the table evaluate_qtable.py gives you).
  2. The quantum layer is swapped for a NumPy simulator (noise_sim.py) that
     implements the same circuit with toggleable noise. With every noise type off, the
     simulator must reproduce the clean Q-table; this is checked at start-up.
  3. For every noise condition and noise seed a noisy Q-table is built (same spike
     encoding every time, so comparisons are paired) and evaluated on the held-out
     layouts. No retraining is involved.

Noise types (choose with --noise; levels with the level flags; 0 / unlisted = off)
  shot      --shots N [N ...]              finite shots per circuit evaluation
  coherent  --coherent_sigma S [S ...]     Gaussian angle error on every CRot (radians)
            --coherent_mode static|dynamic
  depol     --depol_2q P [P ...]           depolarizing prob. per wire after each CRot;
            --depol_ratio R                1-qubit error = depol_2q / R (default R=10)
  readout   --readout_error P [P ...]      symmetric measurement bit-flip probability

By default each enabled type is swept ALONE over its levels (isolated effects). With
--combined the enabled types are applied together, pairing the i-th level of each list.

Examples (add the model flags you trained with; run from the repo root)
  python evaluate_noise.py --checkpoint CKPT.pth --quantum --qubits 8 --type mean \
      --max_freq 100 --num_timesteps 10 --dt 0.1 --mem_tau 0.04 --syn_tau 0.01 \
      --num_static 6 --num_dynamic 1 --grid_size 20 --mask_invalid \
      --noise shot,coherent,depol --shots 256 1024 4096 \
      --coherent_sigma 0.02 0.05 0.1 --depol_2q 0.003 0.01 0.03 --noise_seeds 0 1 2 3 4

Outputs (in --out): noise_per_seed.csv, noise_summary.csv, noise_summary.json
"""
import csv
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

import evaluate_qtable as eq
from noise_sim import NOISE_TYPES, NoiseConfig, NoisyQuantumSim

METRICS = ["SR", "SPL", "PL_m", "TR", "agree", "mean_abs_dQ"]


# --------------------------------------------------------------------------
# Arguments and conditions
# --------------------------------------------------------------------------
def make_parser():
    p = eq.make_parser()
    g = p.add_argument_group("noise")
    g.add_argument("--noise", type=str, default="none",
                   help="comma list of noise types to enable: shot,coherent,depol,readout "
                        "(or 'all' / 'none').")
    g.add_argument("--shots", type=int, nargs="+", default=[1024])
    g.add_argument("--coherent_sigma", type=float, nargs="+", default=[0.05])
    g.add_argument("--coherent_mode", choices=["static", "dynamic"], default="static")
    g.add_argument("--depol_2q", type=float, nargs="+", default=[0.01])
    g.add_argument("--depol_ratio", type=float, default=10.0,
                   help="2-qubit error / 1-qubit error (hardware-like default: 10).")
    g.add_argument("--readout_error", type=float, nargs="+", default=[0.02])
    g.add_argument("--combined", action="store_true",
                   help="apply all enabled noise types together instead of one at a time.")
    g.add_argument("--noise_seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="noise realizations per condition (only used by stochastic noise).")
    g.add_argument("--skip_check", action="store_true",
                   help="skip the noise-free simulator vs PennyLane agreement check.")
    return p


def parse_types(text):
    text = (text or "none").strip().lower()
    if text in ("none", ""):
        return []
    if text == "all":
        return list(NOISE_TYPES)
    types = [t.strip() for t in text.split(",") if t.strip()]
    bad = [t for t in types if t not in NOISE_TYPES]
    if bad:
        sys.exit(f"[error] unknown noise type(s) {bad}; choose from {list(NOISE_TYPES)}")
    return types


def _cfg_kwargs(kind, level, args):
    if kind == "shot":
        return {"shots": int(level)}
    if kind == "coherent":
        return {"coherent_sigma": float(level), "coherent_mode": args.coherent_mode}
    if kind == "depol":
        return {"depol_2q": float(level), "depol_1q": float(level) / args.depol_ratio}
    if kind == "readout":
        return {"readout_error": float(level)}
    raise ValueError(kind)


def _label(kind, level):
    return {"shot": f"shots={int(level)}", "coherent": f"coherent={level:g}",
            "depol": f"depol_2q={level:g}", "readout": f"readout={level:g}"}[kind]


def build_conditions(args):
    """-> list of dicts {label, kind, level, kwargs}."""
    types = parse_types(args.noise)
    levels = {"shot": args.shots, "coherent": args.coherent_sigma,
              "depol": args.depol_2q, "readout": args.readout_error}
    conds = []
    if args.combined and types:
        m = max(len(levels[t]) for t in types)
        for t in types:
            if len(levels[t]) not in (1, m):
                sys.exit(f"[error] --combined needs equal-length level lists (or one value); "
                         f"'{t}' has {len(levels[t])}, others up to {m}.")
        for i in range(m):
            kwargs, parts = {}, []
            for t in types:
                lv = levels[t][i if len(levels[t]) > 1 else 0]
                kwargs.update(_cfg_kwargs(t, lv, args))
                parts.append(_label(t, lv))
            conds.append({"label": "+".join(parts), "kind": "combined", "level": i,
                          "kwargs": kwargs})
    else:
        for t in types:
            for lv in levels[t]:
                conds.append({"label": _label(t, lv), "kind": t, "level": float(lv),
                              "kwargs": _cfg_kwargs(t, lv, args)})
    return conds


# --------------------------------------------------------------------------
# Study loop (model-independent: takes callables, so it can be tested without torch)
# --------------------------------------------------------------------------
def _mean_sd(vals):
    v = np.asarray([x for x in vals if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def run_study(args, Q_clean, make_noisy_qtable):
    conds = build_conditions(args)
    base = eq.summarize(eq.evaluate(Q_clean, args))
    per_seed = [{"condition": "clean", "kind": "clean", "level": 0.0, "seed": "-",
                 "SR": base["SR"], "SPL": base["SPL"], "PL_m": base["PL_m"],
                 "TR": base["TR"], "agree": 1.0, "mean_abs_dQ": 0.0, "seconds": 0.0}]
    print(f"[clean] SR={base['SR']:.3f} SPL={base['SPL']:.3f} "
          f"PL={base['PL_m']:.2f} TR={base['TR']:.3f}")
    clean_arg = Q_clean.argmax(-1)

    for cond in conds:
        cfg0 = NoiseConfig(**cond["kwargs"])
        seeds = list(args.noise_seeds) if cfg0.is_stochastic() else list(args.noise_seeds)[:1]
        for seed in seeds:
            cfg = replace(cfg0, seed=int(seed))
            t0 = time.time()
            Q = make_noisy_qtable(cfg)
            s = eq.summarize(eq.evaluate(Q, args))
            row = {"condition": cond["label"], "kind": cond["kind"], "level": cond["level"],
                   "seed": int(seed), "SR": s["SR"], "SPL": s["SPL"], "PL_m": s["PL_m"],
                   "TR": s["TR"], "agree": float(np.mean(Q.argmax(-1) == clean_arg)),
                   "mean_abs_dQ": float(np.mean(np.abs(Q - Q_clean))),
                   "seconds": time.time() - t0}
            per_seed.append(row)
            print(f"[{cond['label']:<28s} seed {seed}] SR={row['SR']:.3f} "
                  f"SPL={row['SPL']:.3f} agree={row['agree']:.3f}  ({row['seconds']:.0f}s)")
    return per_seed, aggregate(per_seed)


def aggregate(per_seed):
    order, groups = [], {}
    for r in per_seed:
        if r["condition"] not in groups:
            groups[r["condition"]] = []
            order.append(r["condition"])
        groups[r["condition"]].append(r)
    clean_sr = groups["clean"][0]["SR"]
    out = []
    for name in order:
        rows = groups[name]
        d = {"condition": name, "kind": rows[0]["kind"], "level": rows[0]["level"],
             "n_seeds": len(rows)}
        for m in METRICS:
            d[m + "_mean"], d[m + "_sd"] = _mean_sd([r[m] for r in rows])
        d["dSR_vs_clean"] = d["SR_mean"] - clean_sr
        out.append(d)
    return out


def print_table(summary):
    print("\n" + "=" * 96)
    print(f"{'condition':<30s}{'n':>3s}{'SR':>14s}{'dSR':>8s}{'SPL':>8s}"
          f"{'PL(m)':>8s}{'TR':>8s}{'agree':>8s}")
    print("-" * 96)
    for d in summary:
        sr = f"{d['SR_mean']:.3f}+/-{d['SR_sd']:.3f}" if d["n_seeds"] > 1 else f"{d['SR_mean']:.3f}"
        print(f"{d['condition']:<30s}{d['n_seeds']:>3d}{sr:>14s}{d['dSR_vs_clean']:>+8.3f}"
              f"{d['SPL_mean']:>8.3f}{d['PL_m_mean']:>8.2f}{d['TR_mean']:>8.3f}"
              f"{d['agree_mean']:>8.3f}")
    print("=" * 96)
    print("SR/SPL/PL/TR: mean over noise seeds (+/- SD across seeds); deterministic noise "
          "(depol, readout) uses one run.\nagree = fraction of the 2,560 states whose "
          "greedy action matches the clean Q-table. dSR = SR - clean SR.")


def save_results(args, per_seed, summary):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "noise_per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)
    with open(os.path.join(args.out, "noise_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "noise_summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2, default=float)
    print(f"[saved] {args.out}/noise_per_seed.csv, noise_summary.csv, noise_summary.json")


# --------------------------------------------------------------------------
# Model plumbing (needs torch / the repo)
# --------------------------------------------------------------------------
def find_quantum_layer(net):
    """Locate the PennyLane layer by its two parameters ('scaling', 'weights')."""
    found = []
    for name, m in net.named_modules():
        params = dict(m.named_parameters(recurse=False))
        if "scaling" in params and "weights" in params:
            found.append((name, m, params["scaling"], params["weights"]))
            continue
        qw = getattr(m, "qnode_weights", None)
        if isinstance(qw, dict) and "scaling" in qw and "weights" in qw:
            found.append((name, m, qw["scaling"], qw["weights"]))
    if len(found) != 1:
        sys.exit(f"[error] expected exactly one quantum layer with 'scaling' and 'weights' "
                 f"parameters, found {len(found)}: {[f[0] for f in found]}")
    name, m, sc, wt = found[0]
    return name, sc.detach().cpu().numpy(), wt.detach().cpu().numpy()


def replace_module(net, name, new):
    parent = net
    parts = name.split(".")
    for p in parts[:-1]:
        parent = parent._modules[p]
    parent._modules[parts[-1]] = new


def make_wrapper(sim):
    import torch

    class NoisyQLayer(torch.nn.Module):
        """Drop-in replacement for the PennyLane layer (evaluation only, no gradients)."""

        def __init__(self, sim):
            super().__init__()
            self.sim = sim

        def forward(self, x):
            z = self.sim.expectations(x.detach().cpu().numpy())
            return torch.as_tensor(z, dtype=torch.float32, device=x.device)

    return NoisyQLayer(sim)


def main(argv=None):
    p = make_parser()
    args = eq.get_args(argv, parser=p)
    if not args.checkpoint:
        sys.exit("[error] noise evaluation needs --checkpoint (it re-runs the network).")
    if not args.quantum:
        sys.exit("[error] pass --quantum: noise is applied to the quantum layer.")

    net = eq.load_net(args)
    enc = eq.make_encoder(args)
    X = eq.encode_all_states(enc)               # one spike realization, reused everywhere
    Q_clean = eq.qtable_from_net(net, X, args.batch_size)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "qtable_clean.npy"), Q_clean)

    name, scaling, weights = find_quantum_layer(net)
    print(f"[noise] quantum layer '{name}': layers={scaling.shape[0]} qubits={scaling.shape[1]}")
    sim = NoisyQuantumSim(scaling, weights)
    eq_module = make_wrapper(sim)
    replace_module(net, name, eq_module)
    net.eval()

    sim.configure(NoiseConfig())
    Q_sim = eq.qtable_from_net(net, X, args.batch_size)
    diff = float(np.max(np.abs(Q_sim - Q_clean)))
    rel = diff / (float(np.max(np.abs(Q_clean))) + 1e-12)
    agree = float(np.mean(Q_sim.argmax(-1) == Q_clean.argmax(-1)))
    print(f"[check] noise-free simulator vs PennyLane: max|dQ|={diff:.2e} "
          f"(relative {rel:.1e}), greedy-action agreement {agree:.4f}")
    if (rel > 1e-3 or agree < 0.999) and not args.skip_check:
        sys.exit("[error] the NumPy simulator does not reproduce the PennyLane circuit, "
                 "so noisy results would not be trustworthy. Send this output to get it "
                 "fixed (or use --skip_check to continue anyway).")

    def make_noisy_qtable(cfg):
        sim.configure(cfg)
        return eq.qtable_from_net(net, X, args.batch_size)

    per_seed, summary = run_study(args, Q_clean, make_noisy_qtable)
    print_table(summary)
    save_results(args, per_seed, summary)


if __name__ == "__main__":
    main()