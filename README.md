# Who Benefits from *Learned* Surge Pricing?

Master's thesis project: **the welfare incidence of learned ride-hailing control under driver heterogeneity.**

This repository began as a curated deep-research corpus on machine learning for ride-hailing (matching, pricing, rebalancing). From the corpus's gap analysis it develops one gap into a thesis: testing whether a *learned* controller that sets prices/matching/rebalancing reproduces the four-way welfare-incidence structure that Castillo (2025, *Econometrica*) measured for surge pricing — total welfare ↑, rider surplus ↑, driver surplus ↓, platform profit ↓ — and whether that incidence depends on the controller's objective.

## The thesis
- **`thesis/thesis.md`** — the thesis (Markdown).
- **`RESEARCH_PLAN.md`** — research questions, design, and honesty constraints.
- **`grounding/`** — literature grounding for the model and calibration (quotes + page refs from the corpus PDFs).

## Code
- **`src/ridehail/`** — the simulator and controllers:
  - `config.py` — all parameters with literature grounding.
  - `geometry.py`, `demand`/OD — grid, travel times, spatio-temporal demand.
  - `drivers.py` — heterogeneous strategic drivers (logit repositioning, reservation-wage entry/exit, beliefs).
  - `matching.py` — nearest-driver-first matching with endogenous pickup/congestion (wild-goose chase).
  - `env.py` — the market MDP and the four-way welfare decomposition (with a consistency identity).
  - `ars.py` — Augmented Random Search controller (compact zone-shared policy); `ppo.py` — PPO (documented to fail here).
  - `baselines.py` — uniform-pricing counterfactual + Besbes-style myopic surge.
  - `evaluate.py` — welfare aggregation and objective scoring.
- **`experiments/`** — `run_cell.py` (one objective × condition × config), `run_all.sh` (full matrix), `aggregate_results.py` (tables/figures), `inspect_policy.py`, `test_invariants.py`, calibration/diagnostic scripts.
- **`results/`** — `data/` (per-cell JSON), `tables/` (Markdown), `figures/` (PNG).

## Reproduce
```bash
pip install numpy scipy pandas matplotlib torch
PYTHONPATH=src python3 experiments/test_invariants.py     # welfare-identity + elasticity checks
ITERS=65 bash experiments/run_all.sh                      # full experiment matrix (~2h, CPU)
PYTHONPATH=src python3 experiments/aggregate_results.py    # tables + figures
```

## Corpus
The original ride-hailing ML literature review (38 papers across 8 threads), its iterative
revisions, the simulated reviewer reports, and the gap analysis are retained at the repository
root (`literature_review_*.md`, `prof_agent_review*.md`, `research_gaps_summary.md`,
`ride-share-deep-research.md`) and in the `Thread_*/` PDF folders.
