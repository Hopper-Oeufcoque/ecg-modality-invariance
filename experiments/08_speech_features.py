"""Experiment 8 — Speech-channel-robustness features for ECG (I3, novel cross-domain).

Borrowed from microphone-channel-robust speech recognition (decades of channel-
invariance research, never applied to ECG). The insight: ECG and speech are both
quasi-periodic, quasi-stationary signals; "channel" in speech (microphone,
recording environment) ≈ "electrode/recording chain" in ECG. Port the toolkit:

  1. Cepstral mean-variance normalization (CMVN) — per-patient baseline subtraction
  2. RASTA filtering — band-pass on cepstral trajectory removes slow electrode drift
  3. fMLLR/i-vector style patient adaptation — per-patient feature normalization

Compute MFCC-like cepstral features on single-lead ECG, apply CMVN + RASTA, feed
to a classifier. Compare to raw waveform (E2 V5 single-lead baseline) on the
watch task.

Scientific question: do speech-channel-robustness features make a single-lead
model more modality-invariant than raw-waveform training? If yes, the speech
toolkit transfers to ECG — a genuine cross-domain novelty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "08_speech_features"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000
N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)
WATCH_CFG = _cfg(seed=None)


# ---------------------------------------------------------------------------
# Speech-channel-robustness feature extraction
# ---------------------------------------------------------------------------

def _mel_filterbank(n_filt, n_fft, fs):
    """Simple mel filterbank."""
    fmin, fmax = 0.5, fs/2
    mel = lambda f: 1127 * np.log1p(f/700)
    inv_mel = lambda m: 700 * (np.exp(m/1127) - 1)
    melpts = np.linspace(mel(fmin), mel(fmax), n_filt+2)
    hzpts = inv_mel(melpts)
    bins = np.floor((n_fft+1) * hzpts / fs).astype(int)
    fb = np.zeros((n_filt, n_fft//2 + 1))
    for i in range(n_filt):
        for j in range(bins[i], bins[i+1]):
            fb[i, j] = (j - bins[i]) / max(1, bins[i+1] - bins[i])
        for j in range(bins[i+1], bins[i+2]):
            fb[i, j] = (bins[i+2] - j) / max(1, bins[i+2] - bins[i+1])
    return fb


_FB_CACHE = {}
def _fb(n_filt, n_fft, fs):
    k = (n_filt, n_fft, fs)
    if k not in _FB_CACHE:
        _FB_CACHE[k] = _mel_filterbank(n_filt, n_fft, fs)
    return _FB_CACHE[k]


def mfcc_ecg(sig, fs=FS, n_filt=20, n_cep=13, frame=200, hop=100, n_fft=256):
    """MFCC-like cepstral features for ECG.

    Framed short-time spectrum -> mel filterbank -> log -> DCT -> cepstral
    coefficients. frame=200ms, hop=100ms (long because ECG is slow-varying).
    """
    n = len(sig)
    frames = []
    for start in range(0, n - frame + 1, hop):
        w = sig[start:start+frame] * np.hanning(frame)
        spec = np.abs(np.fft.rfft(w, n_fft))**2
        mel = _fb(n_filt, n_fft, fs) @ spec
        mel = np.log(mel + 1e-10)
        # DCT-II for cepstral coefficients
        cep = np.zeros(n_cep)
        for k in range(n_cep):
            cep[k] = np.sum(mel * np.cos(np.pi * k * (np.arange(n_filt)+0.5) / n_filt))
        cep *= np.sqrt(2.0 / n_filt)
        frames.append(cep)
    if not frames:
        return np.zeros((1, n_cep))
    return np.stack(frames)  # (T_frames, n_cep)


def cmvn(cep):
    """Cepstral mean-variance normalization — per-patient baseline subtraction."""
    return (cep - cep.mean(0, keepdims=True)) / (cep.std(0, keepdims=True) + 1e-9)


def rasta_filter(cep, fs_frames=10.0):
    """RASTA band-pass on cepstral trajectory (removes slow electrode drift).

    Classic RASTA: IIR band-pass 0.2659..3.7322 Hz on the cepstral time-trajectory,
    realized as [1 -z^-1] / [1 -0.98 z^-1] then a 4-tap FIR. Simplified here as a
    differencing + first-order low-pass on each cepstral coefficient over time.
    """
    out = np.zeros_like(cep)
    for k in range(cep.shape[1]):
        x = cep[:, k]
        # differencing (high-pass, removes DC/slow drift)
        d = np.concatenate([[x[0]], np.diff(x)])
        # first-order low-pass (smoothing)
        y = np.zeros_like(d); a = 0.98
        y[0] = d[0]
        for i in range(1, len(d)):
            y[i] = a * y[i-1] + (1-a) * d[i]
        out[:, k] = y
    return out


def extract_speech_features(sig, fs=FS, apply_cmvn=True, apply_rasta=True):
    cep = mfcc_ecg(sig, fs)
    if apply_cmvn:
        cep = cmvn(cep)
    if apply_rasta:
        cep = rasta_filter(cep)
    # pad/truncate to fixed frame count
    n_frames = 50
    if cep.shape[0] >= n_frames:
        cep = cep[:n_frames]
    else:
        cep = np.concatenate([cep, np.zeros((n_frames - cep.shape[0], cep.shape[1]))], 0)
    return cep  # (50, 13)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CepstralClassifier(nn.Module):
    """1D CNN over cepstral frames (T=50, C=13)."""
    def __init__(self, n_cep=13, n_classes=N_CLASSES, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_cep, base_ch, 5, padding=2), nn.BatchNorm1d(base_ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch, base_ch*2, 5, padding=2), nn.BatchNorm1d(base_ch*2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(base_ch*2, n_classes),
        )
    def forward(self, x):  # x: (B, n_cep, T)
        return self.net(x)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SpeechFeatDataset(Dataset):
    """Cepstral features from sim-watch Lead-I (domain-targeted, matches E17)."""
    def __init__(self, records, labels_idx, apply_cmvn=True, apply_rasta=True,
                 use_raw_wave=False, n_leads=1):
        self.records = records; self.labels_idx = labels_idx
        self.apply_cmvn = apply_cmvn; self.apply_rasta = apply_rasta
        self.use_raw_wave = use_raw_wave; self.n_leads = n_leads

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        x = rec["ecg"][:SIGLEN]
        if x.shape[0] < SIGLEN:
            x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, WATCH_CFG, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        m1 = leadI.mean(); s1 = leadI.std() + 1e-6
        leadI = (leadI - m1) / s1
        if self.use_raw_wave:
            return (torch.from_numpy(leadI[None].astype(np.float32)),
                    torch.tensor(self.labels_idx[i]))
        cep = extract_speech_features(leadI, FS, self.apply_cmvn, self.apply_rasta)
        return (torch.from_numpy(cep.T.astype(np.float32)),
                torch.tensor(self.labels_idx[i]))


def train(dataset_cls, records, labels_idx, model, epochs=20, lr=1e-3, batch_size=64, tag="m"):
    ds = dataset_cls(records, labels_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def eval_cep(model, records, labels_idx, dataset_cls, batch_size=64):
    model.eval()
    ds = dataset_cls(records, labels_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_l=[]; all_y=[]
    for xb, yb in dl:
        all_l.append(model(xb).numpy()); all_y.append(yb.numpy())
    return np.concatenate(all_l), np.concatenate(all_y)


def per_class_auroc(logits, y, n_classes=N_CLASSES):
    from sklearn.metrics import roc_auc_score
    aucs=[]
    for c in range(n_classes):
        yb=(y==c).astype(int)
        if yb.sum()==0 or yb.sum()==len(yb): aucs.append(float("nan")); continue
        try: aucs.append(float(roc_auc_score(yb, logits[:,c])))
        except Exception: aucs.append(float("nan"))
    return aucs
def macro_auroc(logits, y):
    aucs=[a for a in per_class_auroc(logits,y) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i={c:i for i,c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r['label']] for r in rs])
    tr,te=splits['train'],splits['test']; ytr,yte=lab(tr),lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    results = {}
    # V1: raw waveform (E17 single-lead+sim reference)
    print("\n=== V1: raw waveform (E17 reference) ===", flush=True)
    class Raw(Dataset):
        def __init__(s,r,l): s.r=r; s.l=l
        def __len__(s): return len(s.r)
        def __getitem__(s,i):
            x=s.r[i]['ecg'][:SIGLEN]
            if x.shape[0]<SIGLEN: x=np.concatenate([x,np.zeros((SIGLEN-x.shape[0],12))],0)
            mu=x.mean(0,keepdims=True); sd=x.std(0,keepdims=True)+1e-6; x=(x-mu)/sd
            out=simulate_watch(x,FS,WATCH_CFG,LEAD_NAMES,rng=np.random.default_rng(None))
            li=out['watch']; li=(li-li.mean())/(li.std()+1e-9)
            return torch.from_numpy(li[None].astype(np.float32)),torch.tensor(s.l[i])
    m1=train(Raw,tr,ytr,ECGResNet1d(1,N_CLASSES),epochs=20,tag="V1-raw")
    lg,_=eval_cep(m1,te,yte,Raw)
    results["V1_raw_waveform"]={"macro_auroc":macro_auroc(lg,yte)}
    print(f"  V1 raw L4: {results['V1_raw_waveform']['macro_auroc']:.4f}", flush=True)

    # V2: cepstral features (no CMVN/RASTA)
    print("\n=== V2: cepstral features (no channel-robustness) ===", flush=True)
    class Cep0(SpeechFeatDataset):
        def __init__(s,r,l): super().__init__(r,l,apply_cmvn=False,apply_rasta=False)
    m2=train(Cep0,tr,ytr,CepstralClassifier(13,N_CLASSES),epochs=20,tag="V2-cep0")
    lg,_=eval_cep(m2,te,yte,Cep0)
    results["V2_cepstral"]={"macro_auroc":macro_auroc(lg,yte)}
    print(f"  V2 cep L4: {results['V2_cepstral']['macro_auroc']:.4f}", flush=True)

    # V3: cepstral + CMVN
    print("\n=== V3: cepstral + CMVN (baseline subtraction) ===", flush=True)
    class CepCMVN(SpeechFeatDataset):
        def __init__(s,r,l): super().__init__(r,l,apply_cmvn=True,apply_rasta=False)
    m3=train(CepCMVN,tr,ytr,CepstralClassifier(13,N_CLASSES),epochs=20,tag="V3-cmvn")
    lg,_=eval_cep(m3,te,yte,CepCMVN)
    results["V3_cepstral_CMVN"]={"macro_auroc":macro_auroc(lg,yte)}
    print(f"  V3 CMVN L4: {results['V3_cepstral_CMVN']['macro_auroc']:.4f}", flush=True)

    # V4: cepstral + CMVN + RASTA (full speech-channel-robustness stack)
    print("\n=== V4: cepstral + CMVN + RASTA (full speech stack) ===", flush=True)
    class CepFull(SpeechFeatDataset):
        def __init__(s,r,l): super().__init__(r,l,apply_cmvn=True,apply_rasta=True)
    m4=train(CepFull,tr,ytr,CepstralClassifier(13,N_CLASSES),epochs=20,tag="V4-full")
    lg,_=eval_cep(m4,te,yte,CepFull)
    results["V4_cepstral_CMVN_RASTA"]={"macro_auroc":macro_auroc(lg,yte)}
    print(f"  V4 full L4: {results['V4_cepstral_CMVN_RASTA']['macro_auroc']:.4f}", flush=True)

    metrics={"n_train":len(tr),"n_test":len(te),"classes":SUPERCLASSES,"variants":results}
    (RESULTS/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=list(results.keys()); vals=[results[n]["macro_auroc"] for n in names]
    cols=["#A0A0A0","#4C72B0","#9378B6","#C44E52"]
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.arange(len(names)); bars=ax.bar(x,vals,color=cols,edgecolor="black",lw=0.5)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v+0.008,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([n.replace('_','\n') for n in names],fontsize=8)
    ax.set_ylabel("Macro AUROC (L4)"); ax.set_ylim(0.3,0.8)
    ax.set_title("E8: speech-channel-robustness features on ECG (cross-domain)")
    plt.tight_layout(); plt.savefig(RESULTS/"speech_features.png",dpi=130); plt.close()


if __name__=="__main__":
    main()
