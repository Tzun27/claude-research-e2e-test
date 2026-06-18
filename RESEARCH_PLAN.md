# Research Plan — Welfare Incidence of Learned Surge Pricing

**Working title.** *Who Benefits from Learned Surge Pricing? The Welfare Incidence of Deep Reinforcement Learning Control under Driver Heterogeneity.*

**Type.** Master's-level CS thesis (refined, falsifiable, scoped; not a PhD-level contribution).

---

## 1. Gap and research question

The repo's gap analysis (`research_gaps_summary.md`) recommends, in order: **Gap A0** (structural alignment of a three-way DRL welfare decomposition with Castillo 2025) and **driver-heterogeneity-as-agent** (corpus-novel: no paper models drivers as a heterogeneous population whose surge response differs by type). This thesis tackles both jointly.

**Structural target — Castillo (2025), *Econometrica* 93(5):1811–1854**, verified from the source PDF:
surge pricing, relative to a uniform-pricing counterfactual in which the platform sets the overall price level, yields (as % of gross revenue):

| component | Castillo sign | Castillo magnitude |
|---|---|---|
| Total welfare | **+** | +2.15% |
| Rider surplus | **+** | +3.57% |
| Driver surplus | **−** | −0.98% |
| Platform profit | **−** | −0.50% |

Plus: riders at all income levels benefit; **among drivers, long-hours drivers are hurt most, especially women.**

**Research questions.**
- **RQ1 (alignment).** Does a deep-RL controller that jointly sets prices, matching, and rebalancing on a market with a heterogeneous, strategically-responding driver population reproduce the *sign structure* [+welfare, +rider, −driver, −platform] of Castillo's decomposition? Does it reproduce the rough *proportions* (rider gain ≫ driver/platform loss)?
- **RQ2 (objective-dependence).** The gap analysis asks whether DRL can "match, exceed, or quietly distort" the welfare incidence. Hypothesis: incidence is not a property of surge per se but of the *controller's objective*. Test profit-max vs throughput-max vs welfare-max controllers; determine which reproduces the Castillo structure and which distorts it.
- **RQ3 (heterogeneous incidence).** Does the loss fall most heavily on the most-engaged (long-hours, low-reservation-wage) drivers, as Castillo finds empirically? What is the mechanism?
- **RQ4 (fairness frontier).** If a driver-earnings-fairness constraint is imposed (the corpus's "fairness as hard constraint" consensus: JDRCL, Jusup 2025), what is the welfare–equity trade-off, and at what efficiency cost can the driver loss be neutralized?

These pass the guide's importance/reasonableness/feasibility test: importance — first DRL result tested against a published structural-equilibrium welfare target, and answers the "distort?" question directly; reasonableness — every thread in the corpus points here; feasibility — a *structural-alignment* test on a calibrated proxy market (NOT a Houston replication, which needs proprietary data), runnable on CPU.

## 2. Method

**Simulator (`src/ridehail/`).** Discrete-time, multi-zone spatial ride-hailing market.
- **Space.** G×G grid (default 4×4 = 16 zones; grounded in Gammelli 2021). Travel time = grid distance in epochs.
- **Time.** Episode = one operating day of T epochs with a bimodal (AM/PM peak) demand profile and shifting OD hotspots (residential↔center) to create spatial+temporal imbalance → surge conditions.
- **Demand (elastic).** Potential requests per (zone,epoch) ~ Poisson(λ). Each rider has value v (WTP), destination from an OD matrix, and takes the ride iff `v − price − α·wait ≥ 0` (outside option 0) → price- and wait-elastic demand.
- **Supply (heterogeneous, strategic).** M drivers, each of a type θ differing in (i) reservation wage ρ_θ (opportunity cost/epoch — full-time/long-hours = low ρ; casual = high ρ), (ii) repositioning responsiveness β_θ (softmax temperature over expected zonal earnings), (iii) home-zone preference. Drivers (a) reposition toward higher expected earnings when idle (adaptive, Wardrop-like spatial equilibrium), and (b) go offline when windowed earnings < ρ_θ (entry/exit re-equilibration, grounded in Hall–Horton–Knoepfle). Drivers are a fixed-rule responsive population (not co-learning) → environment stationary for the platform; matches Castillo's "platform optimizes, drivers in equilibrium response."
- **Matching.** Batch assignment (Hungarian/greedy) per epoch on net-value scores using prices (+ optional learned dispatch value), within pickup radius. Delegated to a combinatorial layer parameterized by the learned prices, following the hybrid learning+optimization template (Enders 2023; Gammelli 2021; Xu 2018).
- **Economics.** Rider pays p; driver receives (1−γ)p; platform keeps γp (commission). Fares = base + per-distance × surge multiplier.

**Welfare accounting (per episode, reported as % of gross revenue to match Castillo).**
- Rider surplus `RS = Σ_served (v − p − α·wait)`.
- Driver surplus `DS = Σ_drivers (earnings − ρ_θ · online_time)`, decomposable by type.
- Platform profit `PP = Σ γp − rebalancing_costs`.
- Total welfare `W = RS + DS + PP`.
- **Consistency check (validation):** `W = Σ_served (v − α·wait − pickup_cost) − Σ ρ_θ·online_time` because prices/commissions are internal transfers that cancel. Must hold to machine tolerance.

**Controller.** PPO (implemented in-repo, CPU). Action per epoch = per-zone surge multipliers ∈ [m_min, m_max] + per-zone rebalancing target distribution (+ optional dispatch value). State = idle-supply distribution, recent demand, time-of-day, recent earnings. **Objective treatments:** profit (γΣp), throughput (#served trips), welfare (W). **Baselines:** uniform pricing (single optimized citywide multiplier — the Castillo counterfactual), no-rebalancing, greedy matching, Besbes-style myopic-local surge.

**Castillo-alignment experiment.** For each objective, compare full spatio-temporal **surge** policy vs the **uniform** counterfactual; decompose ΔW, ΔRS, ΔDS, ΔPP (% of GR); compare sign pattern + proportions to Castillo. Decompose ΔDS by driver type (RQ3). Then the fairness-constrained frontier (RQ4) and sensitivity sweeps (elasticity, supply tightness, heterogeneity spread) to establish **scope conditions** — when the Castillo structure holds and when it breaks.

## 3. Deliverables
- `src/ridehail/` — simulator + PPO + baselines + welfare accounting (with unit/consistency tests).
- `experiments/` — scripts for each RQ; configs; seeds.
- `results/` — data tables + figures.
- `thesis/thesis.md` — the full Master's thesis in markdown.
- Adversarial review by multiple advisor subagents; revise to convergence.

## 4. Workflow
1. ✅ Repo understanding; gap pick; verify Castillo target.
2. Grounding (parallel subagents): Castillo mechanism; driver-behavior models; simulator calibration.
3. Build simulator + welfare accounting + tests.
4. Build PPO + baselines; pilot end-to-end.
5. Full experiments (objectives × surge/uniform × fairness × sensitivity × seeds).
6. Figures/tables.
7. Write thesis.
8. Adversarial advisor review → revise → converge.
9. Commit + push to `claude/gallant-galileo-n9efdl`.

## 5. Honesty constraints (the repo's reviewers are harsh — earn the claims)
- This is **structural alignment on a calibrated proxy**, not a Houston replication. Say so everywhere.
- Every reported number traceable to a script + seed; no hand-typed results.
- Welfare consistency check must pass before any welfare claim.
- Report averages with confidence intervals over seeds, not cherry-picked peaks.
- State scope conditions and threats to validity explicitly (sim-to-real gap, single calibration, driver-model assumptions).
