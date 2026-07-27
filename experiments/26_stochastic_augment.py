"""Experiment 26 — Stochastic AW augmentation ("diversity > fidelity", 5-seed).

E25b lesson: the FAITHFUL Phase-A spectral generator was NEUTRAL vs clean
(Δ−0.005), while the CRUDE heavy sim gained the most (+0.049). The utility of
synthetic data came from augmentation DIVERSITY / noise-robustness, NOT from
looking like the target. E26 tests that hypothesis directly: replace the
deterministic transfer with LABEL-PRESERVING STOCHASTIC augmentation
(StochasticAWAugmenter: contact dropouts, motion bursts, dry-electrode gain
wander, variable baseline, mild noise) and see whether it beats clean Lead-I on
real CinC.

Same harness as E25b (binary NORM/AF, PTB-XL Lead-I -> real CinC proxy, 5 seeds,
mean±std, paired deltas). Variants:
  V1 clean Lead-I (bar)
  V2 old heavy sim (reference winner from E25b)
  V3s stochastic-augmented ALONE
  V4s clean + stochastic-augmented (augmentation cocktail)
  V5 oracle real

Acceptance: does Δ(V3s − V1) and/or Δ(V4s − V1) come out clearly positive across
seeds — i.e. does stochastic augmentation actually clear the clean bar where the
faithful generator did not?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig
from src.aw_generator import StochasticAWAugmenter
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "26_stochastic_augment"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


def run_seed(seed, tr, tr_leadI, tr_y, cinc):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx)
    cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    # stochastic augmenter: 2x expansion of the clinical Lead-I set (diverse copies)
    aug = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.0, seed=seed)
    stoch_sigs = [aug.generate(s) for s in tr_leadI]
    stoch_sigs2 = [aug.generate(s) for s in tr_leadI]  # second diverse pass for the cocktail

    # old heavy sim (E25b reference)
    sim_cfg = WatchSimConfig(FS, fs_watch=FS, seed=None)
    sim_sigs = []
    for r in tr:
        x = r["ecg12"][:SIGLEN]
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros((SIGLEN-x.shape[0],12))],0)
        mu=x.mean(0,keepdims=True); sd=x.std(0,keepdims=True)+1e-6; x=(x-mu)/sd
        o = simulate_watch(x, FS, sim_cfg, LEADS, rng=np.random.default_rng(None))
        s=o["watch"]; sim_sigs.append(((s-s.mean())/(s.std()+1e-6)).astype(np.float32))

    out = {}
    def run(tag, sigs, ys):
        m = ECGResNet1d(1, N_CLASSES).to(DEVICE)
        e25.train(m, e25.SigDataset(sigs, ys), epochs=20, tag=f"s{seed}-{tag}")
        auc, _ = e25.evaluate(m, test_sigs, test_y)
        out[tag] = auc
        print(f"  seed {seed} {tag}: {auc:.4f}", flush=True)
    run("V1_clean", tr_leadI, tr_y)
    run("V2_oldsim", sim_sigs, tr_y)
    run("V3s_stochastic", stoch_sigs, tr_y)
    run("V4s_clean_stoch", tr_leadI + stoch_sigs2, tr_y + tr_y)
    m5 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(m5, e25.SigDataset([r["sig"] for r in cinc_ref], [r["y"] for r in cinc_ref]), epochs=20, tag=f"s{seed}-V5")
    auc, _ = e25.evaluate(m5, test_sigs, test_y); out["V5_oracle"] = auc
    print(f"  seed {seed} V5_oracle: {auc:.4f}", flush=True)
    return out


def main():
    print("Loading PTB-XL binary AF/NORM ...", flush=True)
    ptb = e25.load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr = ptb["train"]; tr_leadI = [r["leadI"] for r in tr]; tr_y = [r["y"] for r in tr]
    print(f"  train={len(tr)}", flush=True)
    print("Loading REAL CinC ...", flush=True)
    cA, cN = e25.load_cinc(n_per_class=700); cinc = cA + cN
    print(f"  CinC={len(cinc)}", flush=True)

    seeds = [0, 1, 2, 3, 4]
    per_seed = {}
    for s in seeds:
        print(f"\n===== SEED {s} =====", flush=True)
        per_seed[s] = run_seed(s, tr, tr_leadI, tr_y, cinc)

    keys = ["V1_clean", "V2_oldsim", "V3s_stochastic", "V4s_clean_stoch", "V5_oracle"]
    agg = {}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}
    v1 = np.array([per_seed[s]["V1_clean"] for s in seeds])
    for k in ["V3s_stochastic", "V4s_clean_stoch", "V2_oldsim"]:
        vk = np.array([per_seed[s][k] for s in seeds]); d = vk - v1
        agg[k]["delta_vs_V1_mean"] = float(d.mean())
        agg[k]["delta_vs_V1_std"] = float(d.std())
        agg[k]["delta_positive_in_seeds"] = int((d > 0).sum())

    print("\n===== AGGREGATE (mean±std over 5 seeds) =====", flush=True)
    for k in keys:
        extra = ""
        if "delta_vs_V1_mean" in agg[k]:
            extra = f"  Δvs_V1={agg[k]['delta_vs_V1_mean']:+.4f}±{agg[k]['delta_vs_V1_std']:.4f} ({agg[k]['delta_positive_in_seeds']}/5 seeds +)"
        print(f"  {k:16s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}{extra}", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds, "n_ptb_train": len(tr), "n_cinc": len(cinc),
        "aggregate": agg, "per_seed": per_seed,
        "acceptance_test": "V3s_stochastic and V4s_clean_stoch delta_vs_V1 > 0 across seeds",
        "hypothesis": "diversity > fidelity: stochastic label-preserving augmentation beats clean where faithful spectral transfer (E25b V3) did not",
        "honesty": ["5 seeds, n=700 CinC (350 test/seed)", "AF/NORM only",
                    "stochastic augmentation is label-preserving by construction (no time-warp/inversion)",
                    "same harness/split as E25b for direct comparability"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, keys)
    print("\nDONE.", flush=True)


def _plot(agg, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["V1 clean\nLead-I", "V2 old\nheavy sim", "V3s STOCHASTIC\naug alone", "V4s clean+\nstochastic", "V5 oracle\nreal"]
    means = [agg[k]["mean"] for k in keys]; stds = [agg[k]["std"] for k in keys]
    cols = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(5), means, yerr=stds, capsize=5, color=cols)
    ax.axhline(means[0], color="#4C72B0", ls="--", lw=1, label=f"clean Lead-I bar ({means[0]:.3f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC on REAL CinC (mean±std, 5 seeds)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("E26: stochastic AW augmentation — does diversity beat clean?")
    ax.legend(fontsize=8, loc="upper left")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "stochastic_validation.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
