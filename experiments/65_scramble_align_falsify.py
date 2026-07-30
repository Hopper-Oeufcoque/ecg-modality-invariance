#!/usr/bin/env python3
"""
E65 (E64-Q2 FALSIFICATION) — is scramble-under-alignment WANDER-SPECIFIC or generic aug?

E64 found scramble UNDER alignment STACKS: scramble_align 0.832 vs align 0.805 (+0.027,
38/40, p=1.4e-14) — first method to exceed the E51 alignment ceiling. The PROPOSED
mechanism: per-epoch phase-scrambling the <1.5 Hz WANDER band stops the encoder leaning
on wander structure, freeing capacity the InfoNCE correspondence term spends on morphology.

ALTERNATIVE (must falsify before claiming the mechanism): the +0.027 is just GENERIC
augmentation regularization on the labeled CE stream — ANY per-epoch perturbation would
stack equally, and "wander band" is post-hoc. Two MATCHED controls kill or confirm it:

  CONTROL A — wrong-band scramble: SAME phase-scramble operator on a NON-wander band
    (1.5–3 Hz, just above wander, below QRS). If this stacks equally → band is irrelevant,
    it's "any-band phase scramble" (partly generic, wander story wrong).
  CONTROL B — generic noise: additive Gaussian noise, MAGNITUDE-MATCHED per-sample to the
    exact RMS perturbation wander-scramble introduces (‖scramble(x)−x‖). If this stacks
    equally → it's generic augmentation regularization, NOT wander-specific.

Arms (train PTB-XL AF/N, test real CinC, 40 seeds — all use E51 joint alignment):
  align              : E51 label-anchored alignment (reference / ceiling)
  scramble_align     : wander-band (<1.5 Hz) phase-scramble on CE stream  [E64 claim]
  hfscramble_align   : non-wander-band (1.5–3 Hz) phase-scramble          [CONTROL A]
  noise_align        : magnitude-matched Gaussian noise                   [CONTROL B]

Read:
  scramble_align > BOTH controls (sig)     -> WANDER-SPECIFIC mechanism CONFIRMED (claim holds)
  scramble_align ~= hfscramble_align        -> any-band scramble; not wander-specific
  scramble_align ~= noise_align             -> GENERIC aug regularization; retract wander story
  controls ~= align (both null)             -> strongest: only the wander band stacks

HONEST FLAGS: CinC finger != AW wrist; AF/N easy; band edges + λ=0.1 a-priori; single
train set; 40 seeds; NOT zero-data (uses alignment pairs). Noise magnitude matched to
scramble's per-sample delta-RMS for a fair generic control.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import importlib.util
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)
_s50 = importlib.util.spec_from_file_location("e50", ROOT / "experiments" / "50_sjlife_align.py")
e50 = importlib.util.module_from_spec(_s50); _s50.loader.exec_module(e50)

RESULTS = ROOT / "results" / "65_scramble_align_falsify"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; LAMBDA = 0.1; TEMP = 0.1
CUT = 1.5                       # wander band upper edge (== E59/E64)
HF_LO, HF_HI = 1.5, 3.0         # non-wander control band (above wander, below QRS ~10-25 Hz)

_sos_low  = butter(4, CUT/(FS/2), btype="low", output="sos")
_sos_band = butter(4, [HF_LO/(FS/2), HF_HI/(FS/2)], btype="band", output="sos")

def _phase_scramble(comp, rng):
    F = np.fft.rfft(comp)
    ph = np.exp(1j*rng.uniform(0, 2*np.pi, size=F.shape)); ph[0] = 1.0
    return np.fft.irfft(np.abs(F)*ph, n=len(comp))

def scramble_wander(x, rng):
    low = sosfiltfilt(_sos_low, x).astype(np.float64)
    return ((x - low) + _phase_scramble(low, rng)).astype(np.float32)

def scramble_hf(x, rng):
    band = sosfiltfilt(_sos_band, x).astype(np.float64)
    return ((x - band) + _phase_scramble(band, rng)).astype(np.float32)

def noise_matched(x, rng):
    """Additive Gaussian noise, std = per-sample RMS of what wander-scramble would perturb.
    Fair generic control: same perturbation ENERGY as the wander-scramble arm, no band structure."""
    low = sosfiltfilt(_sos_low, x).astype(np.float64)
    delta = _phase_scramble(low, rng) - low       # exactly what scramble_wander changes
    rms = float(np.sqrt(np.mean(delta**2))) + 1e-9
    return (x + rng.normal(0.0, rms, size=len(x))).astype(np.float32)

AUG = {"wander": scramble_wander, "hf": scramble_hf, "noise": noise_matched, "none": None}

def _norm(x):
    x = np.asarray(x, np.float64)
    if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN-x.shape[0])])
    x = x[:SIGLEN]
    return ((x - x.mean())/(x.std()+1e-6)).astype(np.float32)

class AugLabeledDS(Dataset):
    """Labeled CE stream for the joint trainer with a pluggable per-epoch augmentation."""
    def __init__(self, sigs, ys, aug="none", seed=0):
        self.sigs=sigs; self.ys=ys; self.fn=AUG[aug]; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.sigs)
    def __getitem__(self,i):
        x=np.asarray(self.sigs[i],np.float64)
        if self.fn is not None: x=self.fn(x,self.rng)
        return torch.from_numpy(_norm(x)[None]), torch.tensor(self.ys[i])

def train_joint_aug(sigs, ys, pairs, aug="none", lam=LAMBDA, epochs=20, lr=1e-3, seed=0, tag="joint"):
    """E51 joint alignment; labeled CE stream augmented per `aug`. InfoNCE on raw pairs (unchanged)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = ECGResNet1d(n_leads=1, n_classes=2)
    proj = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32))
    lab_dl = DataLoader(AugLabeledDS(sigs, ys, aug=aug, seed=seed), batch_size=64, shuffle=True, drop_last=True)
    pair_ds = e50.PairContrastDS(pairs, seed=seed)
    pair_dl = DataLoader(pair_ds, batch_size=64, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(list(m.parameters())+list(proj.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); proj.train(); pit = iter(pair_dl); tot=0.0; nb=0
        for xb, yb in lab_dl:
            try: xc, xa = next(pit)
            except StopIteration:
                pit = iter(pair_dl); xc, xa = next(pit)
            opt.zero_grad()
            loss_ce = ce(m(xb), yb)
            zc = proj(e50.encode(m, xc)); za = proj(e50.encode(m, xa))
            loss = loss_ce + lam*e50.info_nce(zc, za, temp=TEMP)
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    return m

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    out={}
    out["align"]            = e25.evaluate(train_joint_aug(tr,tr_y,pairs,aug="none",   seed=seed,tag=f"s{seed}-align"),ts,ty)[0]
    out["scramble_align"]   = e25.evaluate(train_joint_aug(tr,tr_y,pairs,aug="wander", seed=seed,tag=f"s{seed}-wander"),ts,ty)[0]
    out["hfscramble_align"] = e25.evaluate(train_joint_aug(tr,tr_y,pairs,aug="hf",     seed=seed,tag=f"s{seed}-hf"),ts,ty)[0]
    out["noise_align"]      = e25.evaluate(train_joint_aug(tr,tr_y,pairs,aug="noise",  seed=seed,tag=f"s{seed}-noise"),ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL AF/NORM + CinC + SJLIFE ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL={len(tr)}  CinC={len(cinc)}  SJLIFE pairs={len(pairs)}  (wander<{CUT}Hz, hf {HF_LO}-{HF_HI}Hz, λ={LAMBDA})", flush=True)

    seeds=list(range(40)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["align","scramble_align","hfscramble_align","noise_align"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (40 seeds) =====")
    for a in arms: print(f"  {a:18s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    n=len(seeds)
    cmp={
      "scramble_vs_align":        pair(A["scramble_align"],A["align"]),          # reproduce E64 Q2
      "hf_vs_align":              pair(A["hfscramble_align"],A["align"]),         # does wrong-band stack?
      "noise_vs_align":           pair(A["noise_align"],A["align"]),             # does generic noise stack?
      "scramble_vs_hf":           pair(A["scramble_align"],A["hfscramble_align"]),# THE band-specificity test
      "scramble_vs_noise":        pair(A["scramble_align"],A["noise_align"]),    # THE generic-aug test
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:22s} Δ={dl:+.3f} wins {w}/{n} p={p:.4f}")

    d_hf,p_hf = cmp["scramble_vs_hf"][0], cmp["scramble_vs_hf"][2]
    d_no,p_no = cmp["scramble_vs_noise"][0], cmp["scramble_vs_noise"][2]
    beats_hf    = (d_hf>0 and p_hf<0.05)
    beats_noise = (d_no>0 and p_no<0.05)
    if beats_hf and beats_noise:      verdict="WANDER-SPECIFIC (claim CONFIRMED)"
    elif not beats_hf and not beats_noise: verdict="GENERIC AUG (wander story RETRACTED)"
    else:                              verdict="PARTIAL (beats one control, not both)"
    print(f"\n  VERDICT: {verdict}")
    print(f"    vs hf-band  Δ={d_hf:+.3f} p={p_hf:.4f} ({'beats' if beats_hf else 'ties'})")
    print(f"    vs noise    Δ={d_no:+.3f} p={p_no:.4f} ({'beats' if beats_noise else 'ties'})")
    print(f"\n  align {A['align'].mean():.3f} | wander {A['scramble_align'].mean():.3f} | hf {A['hfscramble_align'].mean():.3f} | noise {A['noise_align'].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,5))
    cols=["#4a7","#2b8a3e","#b5651d","#8a6d3b"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.006,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["align"].mean(),color="#4a7",ls=":",lw=1,label="alignment reference")
    ax.set_ylim(0.74,0.86); ax.set_ylabel("AUROC real CinC (40 seeds)")
    ax.set_title(f"E65 — is scramble-under-alignment wander-specific?\n{verdict}   (vs-hf Δ={d_hf:+.3f} p={p_hf:.3f} · vs-noise Δ={d_no:+.3f} p={p_no:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS/"scramble_align_falsify.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'scramble_align_falsify.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":n,"wander_cut_hz":CUT,"hf_band_hz":[HF_LO,HF_HI],"lambda":LAMBDA,"temp":TEMP,
                   "verdict":verdict,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy","band edges + lambda a priori",
                              "noise magnitude-matched to scramble delta-RMS (fair generic control)",
                              "NOT zero-data (uses alignment pairs)","single train set","40 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
