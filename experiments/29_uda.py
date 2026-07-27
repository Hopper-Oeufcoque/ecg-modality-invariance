"""Experiment 29 — Unsupervised Domain Adaptation to close the 0.73->0.93 gap.

The V5 oracle (train on real single-lead) hits 0.93; our best clinical-trained
recipe (E26 clean+stochastic) hits 0.73. That 0.20 is PURE recording-modality
shift (single-lead vs single-lead — NOT lead-count/spatial loss, which is
unrecoverable). It should therefore be attackable with domain adaptation that
uses UNLABELED real target data (which a practitioner realistically has: lots of
unlabeled Apple Watch recordings, few labeled ones).

This experiment stacks UDA methods ON TOP of the E26 winning recipe (source =
clinical Lead-I + stochastic augmentation) and asks: does aligning to the
unlabeled real target move us toward the oracle?

Arms (1-lead ECGResNet1d, 5 seeds, tested on held-out real CinC):
  A0 ERM clean Lead-I                      (baseline, ~0.68)
  A1 ERM clean+stochastic (E26 recipe)     (current best, ~0.73)
  A2 A1 + AdaBN  (recompute BN stats on unlabeled target ref; ~free post-hoc)
  A3 A1 source + Deep CORAL  (align source/target feature covariance)
  A4 A1 source + DANN        (domain-adversarial gradient reversal)
  V5 oracle (real->real)                   (ceiling, ~0.93)

UDA uses cinc_ref SIGNALS ONLY (no labels) for adaptation; eval on cinc_test.
No target labels are ever used by A2/A3/A4 — honest unsupervised adaptation.

Honesty: 5 seeds, n=700 CinC (350 test/seed), AF/NORM only, CinC is the
E6c-validated AW proxy (dist 0.247) not real AW; agreement of features != clinical
validity. Adaptation is to CinC-proxy distribution, not real AW directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d
from src.aw_generator import StochasticAWAugmenter
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "29_uda"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"


# ---- feature access on ECGResNet1d (features = everything before final Linear) ----
def forward_feat(m, x):
    h = m.stem(x); h = m.blocks(h)
    for layer in list(m.head)[:-1]:   # BN, ReLU, AdaptiveAvgPool1d, Flatten, Dropout
        h = layer(h)
    feat = h
    logits = m.head[-1](feat)
    return feat, logits


def _to_tensor(sigs):
    arr = []
    for x in sigs:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN - x.shape[0])])
        x = x[:SIGLEN]; x = (x - x.mean()) / (x.std() + 1e-6)
        arr.append(x[None, :].astype(np.float32))
    return torch.from_numpy(np.stack(arr))


def coral_loss(fs, ft):
    """Deep CORAL: squared Frobenius distance between feature covariances."""
    d = fs.size(1)
    fs = fs - fs.mean(0, keepdim=True); ft = ft - ft.mean(0, keepdim=True)
    cs = (fs.t() @ fs) / (fs.size(0) - 1 + 1e-6)
    ct = (ft.t() @ ft) / (ft.size(0) - 1 + 1e-6)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def train_erm(model, ds, epochs=20, tag="m"):
    return e25.train(model, ds, epochs=epochs, tag=tag)


def train_uda(model, src_sigs, src_y, tgt_sigs, method, epochs=20, tag="uda", seed=0):
    """Train classifier on labeled source + align to unlabeled target (CORAL/DANN)."""
    torch.manual_seed(seed)
    Xs = _to_tensor(src_sigs); ys = torch.tensor(src_y, dtype=torch.long)
    Xt = _to_tensor(tgt_sigs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    dom_clf = None
    if method == "dann":
        dom_clf = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        opt = torch.optim.Adam(list(model.parameters()) + list(dom_clf.parameters()), lr=1e-3)
    bs = 64
    n = Xs.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n); tperm = torch.randperm(Xt.size(0))
        tot = 0.0; nb = 0
        p = ep / max(epochs - 1, 1)
        lamb = 2.0 / (1.0 + np.exp(-5 * p)) - 1.0   # DANN schedule; also CORAL weight ramp
        for i in range(0, n, bs):
            si = perm[i:i + bs]
            xb = Xs[si]; yb = ys[si]
            ti = tperm[(i // bs * bs) % Xt.size(0):][:xb.size(0)]
            if ti.numel() < xb.size(0):
                ti = tperm[:xb.size(0)]
            xt = Xt[ti]
            opt.zero_grad()
            fs, logits = forward_feat(model, xb)
            loss = ce(logits, yb)
            ft, _ = forward_feat(model, xt)
            if method == "coral":
                loss = loss + lamb * 10.0 * coral_loss(fs, ft)
            elif method == "dann":
                feats = torch.cat([fs, ft], 0)
                dom = torch.cat([torch.zeros(fs.size(0)), torch.ones(ft.size(0))]).long()
                rev = GradReverse.apply(feats, lamb)
                dloss = ce(dom_clf(rev), dom)
                loss = loss + dloss
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f} lamb={lamb:.2f}", flush=True)
    return model


@torch.no_grad()
def adabn(model, tgt_sigs, passes=3):
    """AdaBN: reset BN running stats, recompute on unlabeled target via train-mode passes."""
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None  # cumulative average
    Xt = _to_tensor(tgt_sigs)
    model.train()
    for _ in range(passes):
        for i in range(0, Xt.size(0), 64):
            model(Xt[i:i + 64])
    model.eval()
    return model


def run_seed(seed, tr_leadI, tr_y, cinc):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    ref_sigs = [r["sig"] for r in cinc_ref]      # UNLABELED target for adaptation
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    aug = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.0, seed=seed)
    stoch = [aug.generate(s) for s in tr_leadI]
    src_sigs = tr_leadI + stoch; src_y = tr_y + tr_y   # E26 winning recipe = source

    out = {}

    # A0 clean baseline
    m0 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train_erm(m0, e25.SigDataset(tr_leadI, tr_y), tag=f"s{seed}-A0")
    out["A0_clean"] = e25.evaluate(m0, test_sigs, test_y)[0]

    # A1 clean+stochastic (current best)
    m1 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train_erm(m1, e25.SigDataset(src_sigs, src_y), tag=f"s{seed}-A1")
    out["A1_clean_stoch"] = e25.evaluate(m1, test_sigs, test_y)[0]

    # A2 A1 + AdaBN (recompute BN stats on unlabeled target) — operate on a copy
    import copy
    m2 = copy.deepcopy(m1); adabn(m2, ref_sigs, passes=3)
    out["A2_adabn"] = e25.evaluate(m2, test_sigs, test_y)[0]

    # A3 CORAL (source clean+stoch, align to unlabeled target)
    m3 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train_uda(m3, src_sigs, src_y, ref_sigs, "coral", tag=f"s{seed}-A3", seed=seed)
    out["A3_coral"] = e25.evaluate(m3, test_sigs, test_y)[0]

    # A4 DANN
    m4 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train_uda(m4, src_sigs, src_y, ref_sigs, "dann", tag=f"s{seed}-A4", seed=seed)
    out["A4_dann"] = e25.evaluate(m4, test_sigs, test_y)[0]

    # V5 oracle
    m5 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train_erm(m5, e25.SigDataset(ref_sigs, [r["y"] for r in cinc_ref]), tag=f"s{seed}-V5")
    out["V5_oracle"] = e25.evaluate(m5, test_sigs, test_y)[0]

    for k, v in out.items():
        print(f"  seed {seed} {k}: {v:.4f}", flush=True)
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
        per_seed[s] = run_seed(s, tr_leadI, tr_y, cinc)

    keys = ["A0_clean", "A1_clean_stoch", "A2_adabn", "A3_coral", "A4_dann", "V5_oracle"]
    agg = {}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}
    a1 = np.array([per_seed[s]["A1_clean_stoch"] for s in seeds])
    for k in ["A2_adabn", "A3_coral", "A4_dann"]:
        vk = np.array([per_seed[s][k] for s in seeds]); d = vk - a1
        agg[k]["delta_vs_A1_mean"] = float(d.mean())
        agg[k]["delta_vs_A1_std"] = float(d.std())
        agg[k]["delta_positive_in_seeds"] = int((d > 0).sum())

    print("\n===== AGGREGATE (mean±std over 5 seeds) =====", flush=True)
    for k in keys:
        extra = ""
        if "delta_vs_A1_mean" in agg[k]:
            extra = f"  Δvs_A1={agg[k]['delta_vs_A1_mean']:+.4f}±{agg[k]['delta_vs_A1_std']:.4f} ({agg[k]['delta_positive_in_seeds']}/5 +)"
        print(f"  {k:16s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}{extra}", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds,
        "aggregate": agg, "per_seed": per_seed,
        "question": "Does UDA (AdaBN/CORAL/DANN) using UNLABELED real target close the 0.73->0.93 gap over the E26 recipe (A1)?",
        "honesty": ["5 seeds, n=700 CinC (350 test/seed)", "AF/NORM only",
                    "UDA uses cinc_ref SIGNALS only (no labels)", "CinC = E6c-validated AW proxy, not real AW",
                    "adaptation target is the proxy distribution"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, keys)
    print("\nDONE.", flush=True)


def _plot(agg, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["A0 clean", "A1 clean\n+stoch", "A2 +AdaBN", "A3 +CORAL", "A4 +DANN", "V5 oracle\nreal"]
    means = [agg[k]["mean"] for k in keys]; stds = [agg[k]["std"] for k in keys]
    cols = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5, color=cols)
    ax.axhline(means[1], color="#55A868", ls="--", lw=1, label=f"E26 recipe bar ({means[1]:.3f})")
    ax.axhline(means[-1], color="#937860", ls=":", lw=1, label=f"oracle ceiling ({means[-1]:.3f})")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC on REAL CinC (mean±std, 5 seeds)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("E29: does unsupervised domain adaptation close the 0.73→0.93 gap?")
    ax.legend(fontsize=8, loc="upper left")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "uda_validation.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
