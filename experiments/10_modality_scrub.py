"""Experiment 10 — Modality-direction scrubbing via Iterative Nullspace
Projection (INLP).

NOVEL CROSS-DOMAIN METHOD: INLP is the workhorse of NLP fairness / bias-removal
(Ravfogel et al. 2020, "Null It Out"), used to scrub protected attributes (gender,
race) from sentence embeddings by iteratively projecting them into the nullspace
of a linear adversary. It has *never* been applied to ECG recording-modality
invariance. Here "protected attribute" = recording modality (clinical 12-lead vs
watch single-lead); "content" = pathology. The hypothesis: the modality signature
(filter response, electrode physics, noise color) lives in a low-rank linear
subspace of a trained encoder's penultimate features, and projecting it out makes
a downstream pathology classifier invariant to the shift *without retraining the
encoder* — the cheapest possible invariance layer, stackable on any frozen model.

Design (all post-hoc on a frozen, lead-masking-trained backbone — the E2 winner):
  1. Train the 12-lead lead-masking backbone (E2 V2, prob=0.5) on PTB-XL train.
  2. Extract 32-dim penultimate features for train+test in BOTH domains:
       clinical = full 12-lead input
       watch   = Lead-I (full-watch sim) zero-padded to 12 channels (the L4 input)
  3. Modality adversary: logistic regression separating clinical vs watch
     train features. Its weight row(s) span the "modality direction(s)".
  4. INLP: iteratively (K rounds) fit the adversary on the projected features
     and project out the new direction. After K rounds, the modality is (ideally)
     unpredictable from features while pathology content is preserved.
  5. Pathology probe (linear, sklearn) — two regimes:
       V1 CROSS-DOMAIN: train on clinical-train features, test on watch-test.
            Does scrubbing close the train-clinical / test-watch gap?
            (the domain-generalization question)
       V2 IN-DOMAIN: train on watch-train, test on watch-test.
            Does scrubbing HURT? (tests modality/pathology entanglement)
  6. Modality-adversary accuracy after K rounds (sanity: should drop toward chance).
  7. Ablate K = 0 (none), 1, 2, 3, 5, 8 to map the scrub-vs-preserve tradeoff.

Honesty flags: single seed, single backbone, 32-dim feature space (INLP can
exhaust it), sim-watch not real-watch (E6 addresses realism separately),
linear probes only (non-linear modality structure would evade INLP).
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
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "10_modality_scrub"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000
N_LEADS = 12
N_CLASSES = len(SUPERCLASSES)
FEAT_DIM = 32  # base_ch = penultimate feature dim
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# Backbone training (reuse E2 V2 lead-masking recipe)
# ---------------------------------------------------------------------------

class AugDataset(torch.utils.data.Dataset):
    def __init__(self, records, labels_idx, lead_mask_prob=0.5):
        self.records = records; self.labels_idx = labels_idx
        self.lead_mask_prob = lead_mask_prob

    def __len__(self): return len(self.records)

    def _fixlen(self, x):
        T = x.shape[0]
        if T >= SIGLEN: return x[:SIGLEN]
        return np.concatenate([x, np.zeros((SIGLEN - T, 12), dtype=x.dtype)], 0)

    def __getitem__(self, i):
        rec = self.records[i]
        x = self._fixlen(rec["ecg"]).copy()
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        if self.lead_mask_prob > 0.0:
            mask = rng.random(N_LEADS) < self.lead_mask_prob
            mask[0] = False
            x[:, mask] = 0.0
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.labels_idx[i])


def train_backbone(records, labels_idx, epochs=20, lr=1e-3, batch_size=64):
    ds = AugDataset(records, labels_idx, lead_mask_prob=0.5)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [backbone] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, records, domain, batch_size=64):
    """Return (feats [N,32], labels [N]) for a given domain.

    domain='clinical': full 12-lead input (L0).
    domain='watch':    Lead-I full-watch sim, zero-padded to 12 ch (L4 input).
    """
    model.eval()
    cfg_watch = _cfg(seed=0)  # deterministic L4 full watch
    # feature = everything in head except the final Linear (index 5)
    def feats(x):
        h = model.stem(x); h = model.blocks(h)
        return model.head[:5](h)  # (B, 32)

    all_f = []; all_y = []
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:SIGLEN]
            if x.shape[0] < SIGLEN:
                x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            if domain == "watch":
                out = simulate_watch(x, FS, cfg_watch, LEAD_NAMES, rng=np.random.default_rng(0))
                leadI = out["watch"]
                leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
                framed = np.zeros((SIGLEN, 12), dtype=np.float32)
                framed[:, 0] = leadI
                x = framed
            elif domain == "clinical":
                pass  # use as-is (12-lead)
            else:
                raise ValueError(domain)
            batch.append(x.astype(np.float32))
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        all_f.append(feats(xb).numpy())
        all_y.append([rec["label_idx"] for rec in records[i:i+batch_size]])
    return np.concatenate(all_f), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# INLP — Iterative Nullspace Projection
# ---------------------------------------------------------------------------

def inlp_project(Z_train_mod, mod_train, Z_apply, n_rounds, lr_C=1.0):
    """Iteratively fit a linear modality adversary and project its direction out.

    Returns: (Z_apply_projected, directions list, modality_acc_per_round).
    Uses logistic-regression weight row as the direction each round.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    Z = Z_apply.copy().astype(np.float64)
    Zt = Z_train_mod.copy().astype(np.float64)
    dirs = []
    mod_accs = []
    for r in range(n_rounds):
        clf = LogisticRegression(max_iter=2000, C=lr_C, solver="lbfgs")
        clf.fit(Zt, mod_train)
        w = clf.coef_[0]  # (32,)
        w = w / (np.linalg.norm(w) + 1e-12)
        dirs.append(w)
        # project out: Z <- Z - (Z w) w^T
        Zt = Zt - np.outer(Zt @ w, w)
        Z = Z - np.outer(Z @ w, w)
        # modality accuracy on TRAIN (should fall)
        mod_accs.append(float(accuracy_score(mod_train, clf.predict(Zt))))
    return Z, dirs, mod_accs


def pathology_probe(Ztr, ytr, Zte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(Ztr, ytr)
    proba = clf.predict_proba(Zte)
    aucs = []
    for c in range(N_CLASSES):
        yb = (yte == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb): aucs.append(float("nan")); continue
        try: aucs.append(float(roc_auc_score(yb, proba[:, c])))
        except Exception: aucs.append(float("nan"))
    aucs = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan"), aucs


def modality_accuracy(Ztr, mod_tr, Zte, mod_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(Ztr, mod_tr)
    return float(accuracy_score(mod_te, clf.predict(Zte)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    tr, te = splits["train"], splits["test"]
    for r in tr: r["label_idx"] = c2i[r["label"]]
    for r in te: r["label_idx"] = c2i[r["label"]]
    ytr = np.array([r["label_idx"] for r in tr])
    yte = np.array([r["label_idx"] for r in te])
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    print("\n=== Training lead-masking backbone (E2 V2) ===", flush=True)
    model = train_backbone(tr, ytr, epochs=20)

    print("\n=== Extracting penultimate features ===", flush=True)
    Z_clin_tr, _ = extract_features(model, tr, "clinical")
    Z_clin_te, _ = extract_features(model, te, "clinical")
    Z_watch_tr, _ = extract_features(model, tr, "watch")
    Z_watch_te, _ = extract_features(model, te, "watch")
    print(f"  clinical train feats {Z_clin_tr.shape}, watch train feats {Z_watch_tr.shape}", flush=True)

    # Modality labels: 0=clinical, 1=watch.  Combine train for adversary.
    Z_mod_tr = np.concatenate([Z_clin_tr, Z_watch_tr], 0)
    mod_tr = np.concatenate([np.zeros(len(Z_clin_tr)), np.ones(len(Z_watch_tr))])

    # Baseline modality predictability (no scrubbing)
    Z_mod_te = np.concatenate([Z_clin_te, Z_watch_te], 0)
    mod_te = np.concatenate([np.zeros(len(Z_clin_te)), np.ones(len(Z_watch_te))])
    base_mod_acc = modality_accuracy(Z_mod_tr, mod_tr, Z_mod_te, mod_te)
    print(f"  baseline modality-adversary acc (train->test): {base_mod_acc:.4f}", flush=True)

    # Baseline pathology probes (K=0, no scrubbing)
    base_cross, _ = pathology_probe(Z_clin_tr, ytr, Z_watch_te, yte)
    base_indom, _ = pathology_probe(Z_watch_tr, ytr, Z_watch_te, yte)
    print(f"  K=0  cross-domain macro AUROC: {base_cross:.4f} | in-domain: {base_indom:.4f}", flush=True)

    results = {
        "baseline_modality_acc": base_mod_acc,
        "K0_cross_domain_auroc": base_cross,
        "K0_in_domain_auroc": base_indom,
        "rounds": [],
    }

    for K in [1, 2, 3, 5, 8]:
        # project the SAME train+test feature matrices through K INLP rounds
        Z_clin_tr_k, dirs, mod_accs = inlp_project(Z_mod_tr, mod_tr, Z_clin_tr, K)
        Z_clin_te_k, _, _ = inlp_project(Z_mod_tr, mod_tr, Z_clin_te, K)
        Z_watch_tr_k, _, _ = inlp_project(Z_mod_tr, mod_tr, Z_watch_tr, K)
        Z_watch_te_k, _, _ = inlp_project(Z_mod_tr, mod_tr, Z_watch_te, K)
        # modality acc after scrubbing
        Z_mod_tr_k = np.concatenate([Z_clin_tr_k, Z_watch_tr_k], 0)
        Z_mod_te_k = np.concatenate([Z_clin_te_k, Z_watch_te_k], 0)
        mod_acc_k = modality_accuracy(Z_mod_tr_k, mod_tr, Z_mod_te_k, mod_te)
        cross_k, _ = pathology_probe(Z_clin_tr_k, ytr, Z_watch_te_k, yte)
        indom_k, _ = pathology_probe(Z_watch_tr_k, ytr, Z_watch_te_k, yte)
        print(f"  K={K}: modality_acc={mod_acc_k:.4f} | cross={cross_k:.4f} | in-domain={indom_k:.4f}", flush=True)
        results["rounds"].append({
            "K": K, "modality_acc": mod_acc_k,
            "cross_domain_auroc": cross_k, "in_domain_auroc": indom_k,
            "train_modality_acc_per_round": mod_accs,
        })

    metrics = {
        "fs": FS, "n_train": len(tr), "n_test": len(te),
        "classes": SUPERCLASSES, "feat_dim": FEAT_DIM,
        "backbone": "ECGResNet1d 12-lead lead-masking prob=0.5 (E2 V2)",
        "inlp": results,
        "honesty": ["single seed", "single backbone", "32-dim feature space (INLP rank-limited)",
                    "sim-watch not real-watch", "linear probes only (nonlinear modality evades INLP)"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ks = [0] + [r["K"] for r in results["rounds"]]
    mod = [results["baseline_modality_acc"]] + [r["modality_acc"] for r in results["rounds"]]
    cross = [results["K0_cross_domain_auroc"]] + [r["cross_domain_auroc"] for r in results["rounds"]]
    indom = [results["K0_in_domain_auroc"]] + [r["in_domain_auroc"] for r in results["rounds"]]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(Ks, cross, "o-", color="#4C72B0", label="cross-domain AUROC (train clinical → test watch)")
    ax1.plot(Ks, indom, "s-", color="#55A868", label="in-domain AUROC (train watch → test watch)")
    ax1.set_xlabel("INLP rounds K (modality directions removed)")
    ax1.set_ylabel("pathology macro AUROC", color="#333")
    ax1.set_ylim(0.4, 1.0)
    ax1.axhline(0.5, color="gray", ls=":", lw=1)
    ax1.axhline(results["K0_in_domain_auroc"], color="#55A868", ls="--", lw=0.8, alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(Ks, mod, "^--", color="#C44E52", label="modality-adversary acc")
    ax2.set_ylabel("modality-adversary accuracy", color="#C44E52")
    ax2.set_ylim(0.4, 1.0)
    ax1.set_title("E10: INLP modality scrubbing — removing the device direction vs preserving pathology")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(RESULTS / "inlp_tradeoff.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
