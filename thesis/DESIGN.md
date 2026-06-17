# Thesis design specification (internal working doc)

> Working title: **Beyond GMV: Welfare Incidence and Context-Dependent Method Selection in Ride-Hailing Control under Heterogeneous Drivers**
>
> This is the engineering/scientific spec that the methodology chapter is written from. It is the contract the code in `thesis/src/` implements.

## 1. Positioning and the two contributions

The repo's gap analysis (`research_gaps_summary.md`) and the user's refinement in `REPORTING_STYLE.md` converge on two corpus-novel, *implementable* openings:

1. **Driver-heterogeneity-as-agent + welfare incidence (the science).** No paper in the 38-paper corpus models drivers as a heterogeneous population whose response to surge differs, and none reports a stakeholder welfare decomposition across a *library* of controllers. We measure *whom each controller serves*.
2. **Context→method selection vs a brute-force oracle (the method).** Honest reframe of the "agent picks the method" idea: brute force IS the oracle; the question is whether observable city context predicts the winning method cheaply. Evaluated by **regret vs oracle** against random / single-best-fixed / hand-rule baselines.

Both are powered by the *same* experiment grid (method-combo × scenario × seed), so they share infrastructure.

### The "flip the Castillo arrow" principle (load-bearing)
Castillo (2025, *Econometrica*) is **never an optimization target** (that would be a category error: his decomposition encodes a known-harmful incidence). Castillo plays three roles:
- (a) **Motivation:** scalar GMV hides welfare redistribution.
- (b) **Simulator validity check:** calibrate the simulator *only* from independent sources, then test whether it reproduces Castillo's *qualitative* incidence (surge ↑ total welfare & rider surplus, ↓ driver surplus, constrained drivers hurt most). All numbers from the simulator.
- (c) **Reporting template:** decompose welfare into rider / driver-by-type / platform.

## 2. Research questions

- **RQ1 (incidence).** Across the matching×pricing×rebalancing design space, how is welfare redistributed among riders, the platform, and *flexible vs constrained* drivers? Do scalar metrics (GMV, service rate) hide the incidence?
- **RQ2 (validity).** Does a simulator calibrated only from independent sources (Cohen et al. 2016 demand elasticity; Caldwell & Oehlsen 2022 heterogeneous supply elasticities) reproduce Castillo's qualitative incidence pattern?
- **RQ3 (selection).** Can a cheap selector predict the welfare- (and GMV-) maximizing method combination from observable city context, and how close to the brute-force oracle does it get (regret), vs random / single-best-fixed / hand-rule?
- **RQ4 (objective divergence).** Does the *oracle method itself* differ when the objective is GMV vs total welfare vs welfare-with-fairness? (If yes: optimizing GMV systematically mis-selects from a welfare standpoint — the category error, demonstrated.)

## 3. Simulator (`thesis/src/simulator.py`)

Time-stepped, zone-based two-sided ride-hailing market.

- **Space.** N×N grid of zones (default 5×5=25). Travel time between zones = Manhattan grid distance × `minutes_per_cell`. Demand hotspots (CBD-centric vs dispersed) controlled by a spatial concentration parameter.
- **Time.** Discrete steps (1 min) over an episode (default ~300 steps ≈ a multi-hour peak window). Temporal demand profile with controllable peakedness.
- **Demand.** Non-homogeneous Poisson per origin zone, rate λ(zone,t). Each request: origin, destination (OD matrix), rider trip-value V ~ distribution. Rider accepts offered price p iff V − p − VOT·E[wait] ≥ 0 (logit smoothing). Price coefficient calibrated so aggregate price elasticity ≈ target (Cohen 2016 ~ −0.5..−0.6).
- **Supply.** Fleet of M drivers, each a *type* ∈ {flexible, constrained} in proportion `flex_frac`. States: offline / idle / to-pickup / on-trip. Heterogeneity has two grounded mechanisms:
  - **(H1) Surge-chasing / spatial flexibility.** Flexible drivers reposition toward high-fare/high-demand zones when idle; constrained drivers stay near a home zone. Operationalizes Castillo's "limited mobility ⇒ can't chase surge" and Cook 2021's "preferences over where/when to work."
  - **(H2) Labor-supply elasticity (extensive margin).** Flexible drivers enter/exit with expected earnings (high elasticity, ~1.0); constrained drivers work ~fixed hours (low elasticity, ~0.3). Grounded in Caldwell & Oehlsen 2022. Produces Hall–Horton–Knoepfle re-equilibration: flexible drivers flood peaks, competing away the surge premium and crowding out constrained drivers.
- **Matching technology.** Pickup time = travel time from assigned driver to rider (endogenous). We *validate* that emergent mean wait vs idle-density follows the square-root law (canonical pickup-distance ∝ idle-density^−1/2).

### Welfare accounting (Castillo-style, all simulator-computed)
- **Rider surplus** RS = Σ_completed (V − price − VOT·wait).
- **Driver surplus** DS = Σ_drivers (earnings − opportunity_cost·online_hours − driving_cost·miles), reported by type.
- **Platform profit** Π = Σ commission·fare.
- **Total welfare** W = RS + DS + Π. Reported as levels and as % change vs flat-pricing baseline (mirrors Castillo "% of gross revenue").
- Also: GMV (Σ fare), service rate, mean wait, driver idle %, fairness (min-type DS, Gini across drivers).

## 4. Method library (the "menu") — 3×3×3 = 27 combos

**Matching M:** M1 greedy nearest-idle; M2 optimal bipartite (Hungarian, max immediate value within radius); M3 value-based (Xu 2018-style TD value table V(zone,t); edge weight = immediate reward + γV(dest,t+τ) − V(origin,t); Hungarian).

**Pricing P:** P1 flat (mult=1); P2 reactive multiplicative surge mult=clip(1+k·(D−S)/S,1,cap) (Besbes "local myopic"); P3 fluid/equilibrium-optimal price internalizing elasticity + supply (OR/econ family).

**Rebalancing R:** R1 none; R2 greedy-to-demand (idle → nearest high (D−S) zone); R3 value-gradient / min-cost-flow toward forecast demand (Gammelli AMoD bi-level, simplified).

(Small families on purpose, so brute force = oracle is tractable and the "why not brute force?" question is answered head-on.)

## 5. City scenarios (context features)

Observable-without-experiment features → scenario generator (`thesis/src/scenarios.py`):
`demand_supply_ratio`, `spatial_concentration`, `temporal_peakedness`, `demand_elasticity`, `flex_frac`, `trip_length_mean`, (optionally `grid_size`). Structured grid + random samples ⇒ ~40–60 scenarios, each with K seeds.

## 6. Oracle, selector, regret (`thesis/src/selection.py`)

- Run all 27 combos × K seeds per scenario → results matrix.
- **Oracle(objective)** = combo with best mean objective per scenario, for objective ∈ {welfare, GMV, welfare+fairness}.
- **Selector:** features → best combo. Genuine generalization via leave-one-scenario-out / grouped CV. Interpretable models (decision tree, k-NN), possibly per-subtask.
- **Regret** = objective(oracle) − objective(selected), normalized to [0,1] within scenario. Baselines: random, single-best-fixed (the "use one method everywhere" / "JPDR everywhere" analogue), hand-rule, oracle (=0).
- **Contribution test:** selector regret ≪ single-best-fixed ⇒ method choice is context-dependent (selection justified). selector ≈ single-best-fixed ⇒ honest negative result (one method suffices) — still valuable.

## 7. Calibration constants (independent sources only; NOT Castillo's welfare numbers)
- Demand price elasticity: target ≈ −0.5..−0.6 (Cohen et al. 2016, NBER w22627).
- Heterogeneous driver labor-supply elasticity: flexible ≈ 1.0, constrained ≈ 0.3 (Caldwell & Oehlsen 2022, as used by Castillo for supply side).
- Platform commission: ~25% (Uber/Lyft public figures).
- Value of time, driving cost, pickup baseline: from transportation/ride-hailing literature (pending calibration subagent; sanity-bounded).

## 8. Deliverables
- `thesis/src/` simulator + methods + scenarios + selection + experiment driver.
- `thesis/results/` results matrices (CSV/parquet), oracle/selector outputs.
- `thesis/figures/` all figures.
- `thesis/THESIS.md` the thesis (5–7 chapters + refs + appendices).
- Adversarial review pass by advisor/reviewer subagents, then revise.

## 9. Runtime budget
27 combos × ~48 scenarios × ~8 seeds ≈ 10k runs. Target <200 ms/run (lean numpy), multiprocessing over 4 cores ⇒ minutes. Value-table TD pre-trained once per scenario and cached. Tune after a timing probe.
