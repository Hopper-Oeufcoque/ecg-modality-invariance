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

## BATCH SUMMARY (N1–N4 complete, 2026-07-27)

| Idea | Experiment | Verdict | One-line outcome |
|------|-----------|---------|------------------|
| N1 unpaired distribution align (CORAL/Sinkhorn) | E57 | ⚠️❌ | pairing irreplaceable — recovers only ⅓ gain, loses to calibration |
| N2 device-adversarial content/style (DANN+GRL) | E58 | ❌ | backfires (unanchored adversary destroys pathology); triangulates N1 |
| N3 cardiac-cycle per-clip test-time adaptation | E61 | ❌ | per-clip TTA hurts; cardiac locking ≈ random shift (framing irrelevant) |
| N4 frequency-band modality dropout | E59 | ✅⚠️ | **band-scramble ties calibration with ZERO watch data — cheapest lever** |
| N5 amplitude/baseline-invariant handcrafted features | E62/E62b | ✅⚠️ | **crushes AF (0.908, timing-specific); collapses 4× on morphology — rhythm-family lever** |
| — real-AW SEX measurement (bonus, Hop's CSV catch) | E60/E60b | ⚠️ | sex morphology transfers near-intact; method gains small (in-band, no gap) |

**Batch takeaways:**
1. **Two keepers:** (N4/band-scramble) a no-data augmentation matching calibration; and
   (N5/invariant-features) the strongest lever that exists **for rhythm-family tasks** — but
   E62b confirms it's timing-specific (collapses on morphology), not general.
2. **Three unpaired-invariance negatives (N1,N2,N3) converge on one lesson:** you cannot
   manufacture modality invariance without *train-time* correspondence. Distribution
   matching, adversarial scrubbing, and test-time self-supervision all fail to reach even
   calibration. The gold signal remains same-patient paired data (E51).
3. **Unifying axis discovered:** a method's gain ∝ how much of the target signal lives in the
   axis it protects (rhythm vs morphology). Calibration (E47), alignment (E53), and invariant
   features (E62/E62b) all obey this. Deploy two-track: **timing→invariant handcrafted features;
   morphology→deep + alignment/calibration.**
4. **Measurement capability unlocked (E60):** we can now put real-Apple-Watch numbers on
   levers. But **sex is a poor discriminator** (modality-robust, in-band); the decisive
   real-AW method test needs a **disease endpoint with out-of-band sensitivity** = real
   disease labels (pending, Hop acquiring from St Jude).

**Ideation list N1–N6 COMPLETE** (N6 cycle-translation deprioritized: GAN-heavy on CPU, synthesis
historically underperformed alignment here — E6b/SelfMIS; left unrun by design).

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

### 🥉 N3 — Per-clip test-time adaptation via cardiac-cycle self-consistency → E61 ❌ DONE
- **RESULT (E61, 10 seeds):** FAILS. Both per-clip TTA arms (BN-affine adapt via
  entropy+consistency) drop below clean floor (beat 0.675, shift 0.678, both 0/10, p<0.01).
  Cardiac locking irrelevant: beat≈shift (Δ−0.003). Tiny-batch BN corruption + entropy-min
  sharpening wrong predictions + R-peak-aligned beats too duplicate to be informative.
  Train-time methods (calibration 0.737) unbeaten. Kills N3.
- **Why:** the realistic deployment scenario is a *single 30 s watch clip*. Novel
  ECG-specific SSL objective: R-peak-align the beats within the clip, enforce embedding
  consistency across beats (a clip is one patient/rhythm → beats should embed alike);
  adapt only a tiny adapter/BN at test, label-free. Stronger than E5's BN-adapt because
  the objective is cardiac-structure-aware, not just batch statistics.
- **CPU:** moderate (per-clip inner loop) — keep adapter tiny.

### N4 — Frequency-band "modality dropout" augmentation → E59 ✅⚠️ DONE
- **RESULT (E59, 20 seeds):** WIN (partial). band_SCRAMBLE (phase-randomize the <1.5 Hz
  band — keep wander power, destroy structure) TIES calibration (0.739 vs 0.742) using
  ZERO watch data + ZERO target profile; beats clean +0.038 (13/20, p=0.006) → cheapest
  lever yet. band_DROP (random attenuation) HURTS (0.666, −0.035) — removing energy strips
  real P-wave/ST content (info-destruction motif). Still ≪ paired 0.807 (shares the ~0.74
  augmentation ceiling). Follow-up queued: does scramble stack with calibration / under paired?
- **Why:** the gap is baseline-wander (out-of-band). Instead of *injecting* wander
  (calibration), randomly **drop/scramble the low-freq band** during training → forces
  reliance on in-band content by construction. Cheap structured augmentation; a different
  mechanism from calibration (removal vs injection). Quick to run.

### N5 — Physiologically-invariant handcrafted features (amplitude/baseline-free) → E62/E62b ✅⚠️ DONE
- **RESULT (E62 AF, E62b morphology, 20 seeds each):** invariant-by-construction features
  (RR dynamics + normalized spectral shape + Hilbert phase; verified bit-identical under the
  8.25×+offset SJLIFE transform) CRUSH AF (handcrafted 0.908 vs deep 0.701, +0.208, var 8× lower)
  — but E62b confirms this is TIMING-SPECIFIC: margin collapses 4× to +0.049 on morphology.
  Rhythm-family lever, not general. Ensembling with the deep model does NOT help (F6
  complementarity claim fails, both tasks). Establishes the timing→handcrafted / morphology→deep
  two-track deployment split.
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
