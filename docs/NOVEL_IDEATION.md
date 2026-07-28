# Novel / Cross-Domain Method Ideation — Exploration Track

**Branch:** `explore/novel-methods` (keeps `main` as the clean consolidated record).
**Goal (unchanged):** leverage abundant clinical ECG data → transfer to real single-lead
/ Apple-Watch, **zero labeled target fine-tuning**, via recording-modality invariance.
**Grounding:** everything here is shaped by the E38–E56 findings — especially:
(a) the gap is baseline-wander + an in-band residual wall (E37/E43); (b) unanchored
invariance destroys pathology → a **label anchor is mandatory** (E48–E51); (c)
same-patient cross-device correspondence is the gold signal (E51/E51b); (d) benefit is
gap-proportional (E44/E56).

This doc catalogs genuinely novel-to-this-project or cross-domain candidate methods and
ranks them for exploration. Numbered experiments append to the main log when they land.

---

## Ranked exploration backlog

### 🥇 N1 — Drop the paired-hardware dependency (unpaired distribution alignment) → E57 ⚠️❌ DONE
- **RESULT (E57, 20 seeds):** unpaired distribution matching recovers only ~⅓ of the
  paired gain and loses to plain calibration. CORAL 0.715 (+0.014 vs clean, null p=0.31),
  Sinkhorn 0.730 (+0.029, borderline p=0.053, = calibration p=0.52); both ≪ paired 0.807
  (0/20 seeds, p<1e-4). **Pairing is irreplaceable — the mechanism is relational
  (same heart / two devices), not distributional (shape of the feature cloud).** This is
  E51b seen from the other side. Consequence: a few hundred same-patient pairs beat a large
  unpaired watch corpus → direct data collection toward PAIRING, not volume.
- **Why it mattered most:** the headline (E51) needs an *unlabeled paired* set (SJLIFE) —
  the rarest ingredient. If we can recover a meaningful fraction of the gain from
  **unpaired** clinical & watch feature clouds (no same-patient correspondence), the
  method becomes vastly more deployable (any pile of unlabeled watch traces would do).
- **Method (novel framing here):** joint `CE(clinical labels) + λ·D(clinical feats,
  watch feats)` where D is a **distribution** distance, not an instance-pairing loss:
  (a) **CORAL** (2nd-moment/covariance matching, from vision DA); (b) **Sinkhorn OT**
  (entropic optimal transport between the two feature minibatches). The CE anchor is
  retained (E48–E51 lesson). Reference upper bound = E51 paired InfoNCE.
- **Open question:** E51b showed *shuffled instance-pairing* → null. But distribution
  matching is a fundamentally different mechanism (aligns marginals, not instances) — so
  it's a genuine open question whether it recovers any of the gain WITHOUT correspondence.
- **CPU:** trivial (32-d features; CORAL is a 32×32 cov, Sinkhorn a 64×64 transport).

### 🥈 N2 — Content/style disentanglement with a device-adversarial style branch → E58 ❌ DONE
- **RESULT (E58, 20 seeds):** FAILS, triangulates E57. Plain DANN (GRL device
  discriminator on whole feature) drops BELOW clean floor (0.661, −0.040, p=0.005) —
  adversarial force destroys pathology (E50 failure mode via a new objective). Content/style
  split (adversary → 8-d style stub, classifier → 24-d content) repairs to neutral (0.707≈clean,
  p=0.66) but adds nothing; both ≪ paired 0.807 (0/20). Two opposite unpaired approaches
  (E57 passive moment/OT → ⅓ gain; E58 active adversarial → zero/negative) both miss calibration.
  Correspondence (E51b) is the mechanism; unpaired invariance can't manufacture it. N2 killed.
- **Why:** directly attacks the **in-band residual wall** (E43/E48 said the remaining
  gap is information-bound for augmentation). Split the latent into `content`
  (label-predictive, device-invariant via gradient-reversal) and `style`
  (device-predictive); classify from `content` only. Different from E10 (post-hoc linear
  INLP, which was neutral) — this is *by-construction* disentanglement WITH the label anchor.
- **Novelty:** DANN/gradient-reversal is from vision DA; the content/style split with a
  *device* discriminator + label anchor for ECG modality is uncommon.
- **CPU:** cheap (one extra small head + GRL).

### 🥉 N3 — Per-clip test-time adaptation via cardiac-cycle self-consistency
- **Why:** the realistic deployment scenario is a *single 30 s watch clip*. Novel
  ECG-specific SSL objective: R-peak-align the beats within the clip, enforce embedding
  consistency across beats (a clip is one patient/rhythm → beats should embed alike);
  adapt only a tiny adapter/BN at test, label-free. Stronger than E5's BN-adapt because
  the objective is cardiac-structure-aware, not just batch statistics.
- **CPU:** moderate (per-clip inner loop) — keep adapter tiny.

### N4 — Frequency-band "modality dropout" augmentation
- **Why:** the gap is baseline-wander (out-of-band). Instead of *injecting* wander
  (calibration), randomly **drop/scramble the low-freq band** during training → forces
  reliance on in-band content by construction. Cheap structured augmentation; a different
  mechanism from calibration (removal vs injection). Quick to run.

### N5 — Physiologically-invariant handcrafted features (amplitude/baseline-free)
- **Why:** SignalMC-MED (F6) claims handcrafted features are *complementary* to learned
  embeddings. Features invariant to amplitude scaling and baseline by construction
  (instantaneous phase, RR-interval dynamics, spectral-coherence structure) as an
  ensemble add-on to the E51 encoder. Cross-domain from classical signal processing.

### N6 — Physiologically-constrained cycle translation (lower priority)
- **Why:** CycleGAN clinical↔watch with an RR-interval/rhythm-preservation constraint.
  Cross-domain (image-to-image translation). **Deprioritized:** GANs are heavy on CPU and
  synthesis historically underperformed alignment here (E6b, SelfMIS warning). Only if
  the cheaper ideas stall.

---

## Exploration protocol
- Same rig as the main arc: PTB-XL Lead-I train, real CinC AF test, ≥10–20 seeds, paired
  stats vs a seed-matched clean baseline; per-record z-score; `n_leads=1`.
- Bar to beat: clean 0.701, calibration 0.742, **paired alignment (E51) 0.807** (the
  reference ceiling for zero-label). A novel method is "interesting" if it beats
  calibration; "important" if it approaches E51 *without the paired requirement* or
  breaks the in-band wall.
- Negatives logged with equal weight (this is a search — most shots miss, and that's data).
- Winners get promoted to `main` + the synthesis doc; the branch keeps the full search.
