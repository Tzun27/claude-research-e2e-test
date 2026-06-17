# Research gaps overview — ride-hailing ML/RL literature (2018–2026)

*Source: `literature_review_ride_hailing_ml_2018_2026_v5.md` (38 papers across 8 threads). Audience: research group meeting.*

---

## The setup in one paragraph

Ride-hailing platforms make three coupled operational decisions on a city-scale spatiotemporal graph: **matching** (whom to assign to whom), **pricing** (what to charge), and **rebalancing** (where idle drivers should go). Across ~38 papers covering deep RL dispatching, graph-MARL, AMoD hybrid learning+optimization, pricing, LLMs as controllers, mean-field game RL (MFG-RL), driver-labor economics, and ST-GNN forecasting, four real research gaps stand out — plus a fifth pattern that is now better described as an emerging consensus.

---

## The four gaps

### Gap A0 — Structural alignment of a three-way DRL welfare decomposition with Castillo 2025 *(load-bearing)*

**The question.** Castillo 2025 (*Econometrica*, Houston/Uber) gave a four-way welfare decomposition of surge pricing: **+2.15% total welfare, +3.57% rider surplus, −0.98% driver surplus, −0.50% platform profits** (all as % of gross revenue), with female and long-hours drivers most hurt. Can a DRL controller that jointly handles matching + pricing + rebalancing *reproduce the structure* of that decomposition (signs and rough proportions across the four parties) on a proxy market?

**Why it matters.** Every preceding theme in the review converges here. It would be the first DRL result directly tested against a published structural-equilibrium target, and it would tell us whether DRL can match, exceed, or quietly distort the welfare incidence the OR model produced.

**What makes it feasible now.** Sun 2024 (TKDE) showed constrained multi-objective MARL converges in the pairwise case; Zhang 2023 showed joint pricing+matching DRL works on a NYC simulator; the MFG-RL thread (Guo 2019, Cui-Koeppl 2021) provides the theoretical scaffolding; Hall-Horton-Knoepfle 2023 and Cook et al. 2021 give the empirical motivation for treating drivers as a heterogeneous strategic population.

**The honest caveat.** Houston Uber data are not public. The realistic version is *NYC taxi data + a Houston-calibrated demand-elasticity and driver-heterogeneity model*. That is a structural-alignment test, not a Houston replication. Budget: 1–2 years.

---

### Gap A — Three-way joint optimization with strategic-agent guarantees

**The question.** No paper in the corpus optimizes matching + pricing + rebalancing *simultaneously* with explicit strategic-driver assumptions. JDRCL (Sun 2024) does two-way joint matching+repositioning under max-min fairness; Zhang 2023 does joint pricing+matching; nobody does all three.

**Two feasible paths.** (a) Couple a fluid-model pricing module to a DRL matching+repositioning module. (b) A single MARL formulation with prices as agent actions on a coarser timescale.

**Counter-reason worth taking seriously.** The OR/economics fluid-model line (Besbes 2021, Afeche 2023) has solved versions of this analytically. The right framing may be "the ML literature should adopt the equilibrium framing rather than reinvent three-way coupling inside MDPs." Gap A0 is the scoped, falsifiable specialization of this broader gap.

---

### Gap B — LLM role taxonomy and shared benchmark for ride-hailing

**The question.** The four LLM/foundation-model papers in the corpus (UniST, UrbanGPT, GARLIC, LLM-ODDR) play *different roles* — forecaster, encoder, reward designer, direct planner — and report on different benchmarks with no common evaluation harness. LLM-ODDR's headline claim of **+5.21% to +48.87% GMV** has a 9× endpoint ratio that should be read skeptically (it's an unreviewed preprint).

**What's needed.** A role-comparison study on a shared benchmark would tell us *which* LLM role is actually the right one before the field anchors on inflated numbers. A frozen benchmark across all four roles is feasibility-constrained for now; a smaller role-comparison study is the right near-term project.

---

### Gap C — Cross-city transfer and sim-to-real for production deployment

**The question.** Encoder-only transfer is partly solved (UniST, UrbanGPT, plus Tang 2019's CVNet on Didi). *Full-pipeline* transfer — taking a trained controller from city A and deploying it in city B — is not. Tresca 2025's macro→meso zero-shot gap (~60%) and the bi-level GNN-RL line's reliance on city-specific simulators expose this.

**Why it matters operationally.** Real platforms expand across cities. The deployment lineage (Xu 2018 → Tang 2019 → Eshkevari 2022 → Azagirre 2024) is essentially seven years of single-platform success reports, all with concurrent-improvement confounds and switchback identification concerns documented in §6.5.

---

## Emerging consensus (not a gap)

**Fairness as a hard constraint.** JDRCL (Sun 2024), Jusup 2025, and Lyu 2025 now treat fairness as a first-class constraint rather than a post-hoc consideration. Cook et al. 2021's 7% gender earnings gap decomposition and Castillo 2025's finding that female and long-hours drivers are most hurt by surge ground this empirically. This is more accurately called a consensus crystallizing than a gap to fill — but it intersects every other gap above.

---

## A novel opening worth flagging (from §8, not §7)

**Driver-heterogeneity-as-agent.** The corpus uses either *drivers/vehicles* or *grids/regions* as agents. **No paper models drivers as a heterogeneous population whose responses to surge differ by demographic** — even though Cook 2021 and Castillo 2025 both document exactly that heterogeneity empirically. A paper that did this would be corpus-novel and would directly slot into Gap A0.

---

## Recommendation for what to tackle

In rough order of "concrete, defensible, novel":

1. **Gap A0** — the most specific, most falsifiable, and the one every other thread points at. Best as a 1–2 year project.
2. **Driver-heterogeneity-as-agent** — smallest scope, corpus-novel, can be a stepping stone toward Gap A0.
3. **Gap B (role-comparison study)** — best near-term contribution if we want to enter the LLM-for-ride-hailing line without overclaiming.
4. **Gap C / Gap A** — important but either field-wide and slow (C) or partly subsumed by A0 (A).

---

*Tell me which gap you want expanded into a slide deck and I'll pull it together — I'll need a 5-min, 15-min, or 30-min target so I size the deck appropriately.*
