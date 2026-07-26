# ECG Modality Invariance — Agent Working Notes

Context auto-loaded for any Hermes session working in this directory.

## Project
Research compilation on transferring clinical ECG AI models to Apple Watch single-lead ECGs, focused on **recording-modality invariance**. Owner: Luke (US-based). Repo: github.com/Hopper-Oeufcoque/ecg-modality-invariance (private).

## Deliverable
A living, well-organized knowledge base (not a shipped model):
1. **Method taxonomy** — every technique, tagged by category (preprocessing / features / architecture / domain-adaptation / lead-synthesis / SSL / novel).
2. **Evidence** — what has been tried in the literature, with citations + reported results.
3. **Frontier** — plausible-but-unexplored methods, incl. ideas borrowed from adjacent fields.

## Domains of shift to address (12-lead clinical → 1-lead watch)
lead count, electrode physics (wet vs dry), noise, sampling/bandwidth, population/context.

## Working conventions
- Primary sources only for load-bearing claims (PubMed, arXiv, IEEE). Cite with DOI/arXiv ID.
- Every method gets an entry in `docs/method_taxonomy.md` with: name, category, mechanism, evidence, watch-relevance, novelty flag.
- Keep a running idea log in `notes/idea_log.md` for novel/unexplored directions.
- Phased work with checkpoint reporting — report at each phase boundary, ask before deviating.

## Datasets worth noting (fill in as found)
- PTB-XL, CODE-15, MIMIC-IV-ECG (clinical 12-lead)
- Apple Heart Study, Apple Watch ECG single-lead sets, PhysioNet single-lead sets
