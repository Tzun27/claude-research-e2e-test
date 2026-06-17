# Literature review: deep RL, GNNs, and LLMs for ride-hailing matching, pricing, and rebalancing (2018–2026) — v2

> Synthesis-mode literature review of 27 peer-reviewed and preprint papers, organized into five threads (production-scale RL dispatching; graph-MARL and mean-field; AMoD hybrid learning + optimization; pricing; LLMs / foundation models / joint optimization). The review is structured around the six-step workflow from `literature_review_guide_for_ai_agents.md`: search → fast-read & tag → synthesis matrix → extract themes → judge gap → MEAL prose. Section 1 frames the question and documents the search strategy. Sections 2 and 3 expose the schema entries and the matrices. Sections 4 through 8 are the synthesis prose, organized by theme rather than by paper. Section 9 is the gap analysis. Section 10 is the audit checklist.
>
> **v2 changelog (responding to peer review).** Section 0 expands the verification note to flag two extraction errors corrected in v2 (Castillo 2025 direction; Yang 2024 cell). Section 1.1 adds an explicit search-strategy and inclusion/exclusion subsection, with five named adjacent literatures bounded out and a defense for each. Section 2 schema entries for Castillo 2025 and Yang 2024 are corrected. The three load-bearing "(unverified)" numbers (Enders 2023, Hoppe 2024, Zhang 2023) are demoted from theme-anchoring to illustrative status in Section 5, and the themes are re-derived from verified anchors. Theme 1's topic sentence is softened; Theme 2 is scoped to "within the corpus reviewed"; Theme 4 is rewritten with the corrected Castillo direction and explicit Houston/Uber scoping, and acknowledges the mean-field-game RL bridge that the v1 controversy claim missed. Section 6 drops the deployed-vs-academic gap surprise and replaces it with the corpus-specific observation that no paper uses the rider as the agent. Section 7 rewrites each gap with at least one counter-reason and a verdict that turns on which side is stronger. Section 11 (new) lists adjacent literatures excluded with seed citations the next-revision corpus should add. The "seminal" label is restricted to genuinely paradigm-defining works.

---

## 0. Source-attribution and verification note

Effect sizes in this review were re-verified against each paper's abstract, conclusion, or experiment table after an early-pass schema extraction produced fabricated numbers in Thread A. The following headline numbers were corrected by direct read in v1: `Xu2018` (0.5–5% revenue, not 16–19%), `Eshkevari2022` (1.3% A/B, up to 5.3% post-deployment, not 7–9%), `Yue2024` (0.7–3.9% TDI / 1–2% CR, not 8–12%), `Zhang2024 NondBREM` (3.76% order response rate, not 10–15%), `Azagirre2024` ($30M/yr Lyft incremental revenue, not 5–8% acceptance), `Hao2023 GAT-MF` (42.7% performance, 86.4% training-time saving, 19.2% GPU saving — three separate numbers across heterogeneous benchmarks, not a single ride-hailing dispatch number).

**Two further extraction errors corrected in v2** (flagged in peer review):

- **`Castillo2025` direction was inverted in v1.** The paper finds that *surge pricing increases* total welfare by 3.53% relative to a uniform-pricing counterfactual (riders +6.97% with surge; drivers −1.97% with surge). v1 reported this as the welfare effect of *eliminating* surge, with the signs accordingly reversed. The schema entry, Matrix 4, and Theme 4's prose have all been re-derived from the corrected direction in v2. This was exactly the failure mode the guide warns about (fluent prose laundering an upstream extraction error), and the fix changes Theme 4's analysis non-trivially: surge has positive aggregate welfare effect with redistribution from drivers to riders, not the reverse.
- **`Yang2024` numbers were sourced from the wrong table cell.** The "0.24–1.43% IORR / 0.25–1.75% IGMV" range cited in v1 is from a different cell (likely City A or a partial-horizon row) than the headline GRC vs. baseline comparison, which on Table 1 City B OPE shows roughly 5–6% IORR/IGMV gains. v2 reports the corrected ~5–6% range and re-derives Theme 1's "production-scale gains are small" claim from a properly anchored set of numbers (Xu2018, Eshkevari2022, Yue2024, Azagirre2024, Zhang2024 NondBREM); the corrected Yang 2024 figures are still on the smaller end of academic-benchmark reports but no longer support the original "<2%" anchor.

**Three numbers remain "(unverified)" because the present review pass could not retrieve the source PDF cleanly:** `Enders2023` 5–17% profit, `Hoppe2024` 2–6% performance, `Zhang2023` +17% revenue and −14% vehicles. The v1 draft used these as theme-anchoring evidence, which the peer review correctly flagged as inappropriate. **In v2, these claims are demoted to illustrative status only**: Theme 1 no longer cites Enders 2023 as the architectural anchor (Wang2025 CoopRide and Hu2025 BMG-Q now play that role), and Theme 4 no longer leans on Zhang 2023 as the sole DRL pricing exemplar (the controversy claim is rewritten to acknowledge that mean-field-game RL — bounded out of this corpus, see Section 11 — is exactly the bridge between OR equilibrium analysis and DRL that v1 declared absent).

---

## 1. The question this review is answering

Ride-hailing platforms make three operational decisions on a spatiotemporal graph of a city: **whom to assign to whom (matching)**, **what to charge (pricing)**, and **where idle drivers should go (rebalancing)**. These decisions are coupled — pricing changes demand, demand shapes matching feasibility, and matching outcomes determine the post-trip idle distribution that rebalancing then has to correct. The question this review is answering is: *what does the 2018–2026 ML and OR literature on these three decisions look like, separately and jointly, and where are the bridges and gaps between them?*

The five threads from `ride-share-deep-research.md` are kept as a coverage device but are not used as the organizing principle of the prose, which is theme-first by design.

### 1.1 Search strategy, inclusion criteria, and bounded-out adjacent literatures

**Databases and time window.** The corpus was assembled by (a) seeding from `ride-share-deep-research.md`, an in-house deep-research compilation; (b) cross-checking against arXiv (2018–April 2026), ACM Digital Library (KDD, CIKM, WWW, AAAI), PMLR (ICML, L4DC), IEEE Xplore (CDC, ECC, T-ITS, TCNS), INFORMS journals (Management Science, MSOM, Journal on Applied Analytics), and Wiley (Econometrica). The default time window is 2018–2026, with explicit carve-outs for landmark older works that seed reviews cite (e.g., Lin 2018, Xu 2018).

**Keyword set.** "Ride-hailing" OR "ride-sourcing" OR "AMoD" OR "autonomous mobility on demand" OR "robo-taxi" — combined with — "deep reinforcement learning" OR "MARL" OR "multi-agent" OR "graph neural network" OR "GNN" OR "foundation model" OR "LLM" OR "large language model" — combined with — "matching" OR "dispatching" OR "rebalancing" OR "repositioning" OR "fleet management" OR "pricing" OR "surge".

**Inclusion criteria.** Peer-reviewed papers in the venues above plus arXiv preprints with claim-level evidence (a clearly stated method, a defined dataset, and a numeric result). Position papers and tutorials were excluded. Where multiple versions of a paper exist (preprint + journal), the most recent peer-reviewed version is cited.

**Corpus size.** 27 papers — **below the guide's own 30–50 target**. The shortfall is acknowledged. The corpus was capped because four search expansions in the directions listed below produced candidate papers that, on first read, would have stretched the review across two distinct subfields rather than deepening the core ride-hailing-control story. Section 11 lists those expansions as concrete additions for v3.

**Bounded-out adjacent literatures (and why).** A senior reviewer in this area would expect the following five literatures to appear; each is bounded out of v2 with a defense, and seed citations are listed in Section 11.

1. **Mechanism-design and incentive-compatible spatial pricing** (Ma, Fang, Parkes and follow-ons). Bounded out because the review's pricing thread is anchored on the welfare/equilibrium question (does spatial pricing improve total surplus and how is it distributed?), and IC mechanism design answers a different question (does the pricing rule incentivize truthful reporting?). The two are complements rather than substitutes; v3 should add an IC-pricing sub-thread to Theme 4.
2. **Mean-field-game-aware reinforcement learning** (Guo et al.; Saldi; Yang Yaodong's MFG-RL line). Bounded out **partly in error** in v1 — this is the literature that bridges OR equilibrium pricing and DRL control, and Theme 4's controversy claim ("DRL pricing is not in dialogue with OR/economics") is therefore overstated. v2's Theme 4 acknowledges this. v3 should treat MFG-RL as a sixth thread.
3. **Ride-pooling / shared rides beyond BMG-Q** (Alonso-Mora et al. 2017 PNAS framework and descendants). Bounded out because pooling adds a capacity-constraint dimension orthogonal to the matching/pricing/rebalancing axes that organize this review. Hu2025 BMG-Q is included because it is RL-side and graph-attention; the optimization-side pooling literature is not.
4. **Driver labor economics** (Hall, Horton, Knoepfle on Uber labor supply; Cook et al. on the Uber gender earnings gap). Bounded out because the corpus is focused on platform-side control, not driver-side empirical labor questions. The strategic-driver assumption that Theme 4 wants DRL to adopt is empirically grounded in this literature, however, and v3 should cite at least Hall-Horton-Knoepfle as motivation.
5. **ST-GNN demand-forecasting backbone** (DCRNN, Graph WaveNet, STGCN). Bounded out because the review's GNN-encoder discussion (Theme 2) is about *control* representations, not forecasting. The forecasting literature feeds in via UniST/UrbanGPT but is not separately threaded.

**Geographic and platform scope.** The empirical portion of the corpus is heavily concentrated on Didi (Xu 2018, Eshkevari 2022, Yang 2024, Yue 2024, Zhang 2024 NondBREM — five of six deployment/empirical papers), Lyft (Azagirre 2024 — one paper), and NYC-taxi-derived simulators (Lin 2018, Hu 2025, Tresca 2026, Han 2025, Lyu 2025, Yuan 2024, Li 2024). Bolt / FreeNow (EU), Ola (India), Grab / Gojek (Southeast Asia), and DiDi-Brazil deployments are not represented. The Castillo 2025 welfare result is from a single market (Houston). Where empirical claims are presented in Sections 4–8, this geographic concentration is flagged and claims are scoped accordingly.

---

## 2. Schema entries (Step 2 product)

The schema below holds, for each paper, the six fields specified in the guide (`population`, `method`, `key_finding`, `effect_size`, `limitations`, `stance`) plus a controlled-vocabulary tag set. Entries are grouped by thread for traceability but are referenced from Section 3 onward by `citation_key` only. Effect sizes that could not be located in the paper's abstract or results are marked `n/a`. Numbers re-extracted in v2 are flagged inline; numbers that remain unverified are flagged `(unverified)` and are not used as theme-anchoring evidence in Section 5. A tag dictionary is provided at the end of this section.

### 2.1 Thread A — production-scale and offline RL for dispatching (6 papers)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Xu2018 | KDD 2018 | Didi platform; >25M daily orders; major Chinese cities | system deployment; offline value-function learning + online Hungarian matching ("learning + planning") | Long-horizon driver-state value learning + Hungarian matching → 0.5–5% revenue improvement at production scale. | 0.5–5% revenue (deployed); paper also notes ~10% completion-rate gain from earlier driver-select→platform-assign shift, not from RL itself | tabular value table, hand-engineered states, no fairness, single-platform | foundational "learning + planning" baseline; reference for almost every later RL dispatcher | #value-function, #bipartite-matching, #didi, #production-deployment, #real-data |
| Eshkevari2022 | KDD 2022 | Didi international markets; full-scale deployment | system deployment; on-policy DRL with online state-value iteration, signal smoothing, MAB-pruned graph | Online state-value iteration → ≥1.3% A/B driver-income gain, up to 5.3% post-deployment on major metrics. | ≥1.3% A/B; up to 5.3% post-deployment (causal-inference attributed) | exploration-exploitation pressure, single-platform, multi-armed-bandit pruning heuristics, dataset-shift not analytically bounded | first standalone RL dispatcher deployed at scale internationally; supports online RL with safeguards | #online-RL, #didi, #production-deployment, #real-data, #safety-exploration |
| Yang2024 | KDD 2024 | Didi-collected real-world data, 3-week city-level pilots; multiple cities | offline RL on logged data, framed as cooperative Markov game (FOVIR); "Goal-Reaching Collaboration" credit assignment; KM matcher | Modeling future-revenue impact of one region's value on others → consistent improvement over neighbor-value baselines; ~5–6% IORR/IGMV in the City B OPE comparison [v2 corrected]. | ~5–6% IORR/IGMV (City B OPE table; v2 corrected — v1 had cited a partial-horizon range of 0.24–1.43%, sourced from a different cell) | OPE variance is high (±7–10pp on per-horizon estimates), goal-state tuning fragile, model-environment mismatch, single-city training per run | contests prior schemes that evaluate region values independently; supports cooperative-game framing of dispatching | #offline-RL, #MARL, #didi, #real-data, #value-function |
| Yue2024 | CIKM 2024 | Didi MICOD setting; geo-fence ~30 km², 40–50 hexagons; calibrated simulator + Didi data | end-to-end DRL (D2SN encoder–decoder) replacing two-stage value+Hungarian pipeline; two-layer MDP | End-to-end o–d assignment generation → 0.7–3.9% TDI, 1–2% CR over best two-stage DRL baseline. | 0.7–3.9% TDI; 1–2% CR (at 1–4 D&S levels) | ms-level latency budget is tight, driver-supply-density extrapolation untested, evaluation only on Didi simulator/data, micro-view scope by construction | contests two-stage value+Hungarian as the necessary architecture; supports end-to-end as a viable alternative | #end-to-end, #didi, #production-deployment, #real-data |
| Zhang2024 (NondBREM) | AAAI 2024 | Real-world ride-hailing dataset, 50,000+ vehicles, 20M orders | offline DRL with Nondeterministic-BCQ + Random Ensemble Mixture | Offline RL with nondeterministic action-space modeling → 3.76% order-response-rate improvement on logged data. | 3.76% order-response-rate improvement | extrapolation error remains the central bottleneck, no deployment numbers, single dataset, ensemble compute cost | supports offline RL as the deployment-safe alternative to online RL | #offline-RL, #real-data, #ensemble |
| Azagirre2024 (Lyft) | INFORMS J. Applied Analytics 2024 (Edelman finalist; arXiv 2023) | Lyft global production; "most" Lyft markets; switchback experimentation | system deployment; online RL for matching with real-time driver-earnings estimation; "first documented" production rideshare RL match | Online matching RL with future-earnings estimation → ≥$30M/yr incremental revenue, "millions of additional riders." | ≥$30M/yr (deployed); rider-served growth qualitative | switchback experimentation gives bounded but coarse causal attribution, single-platform, fairness/strategic-driver effects not isolated | first non-Didi deployed RL match; supports online RL outside the Didi ecosystem | #online-RL, #lyft, #production-deployment, #matching, #real-data |

### 2.2 Thread B — graph-based and mean-field MARL (8 papers)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Lin2018 | KDD 2018 | Large-scale ride-hailing simulator (NYC taxi-derived) | contextual MARL (cDQN, cA2C) for fleet management | Contextual MARL with shared global state → outperforms rule-based and DQN baselines on order-response-rate. | n/a (relative gain only) | simulator-only, fixed agent vocabulary, no real-data deployment | early MARL repositioning baseline | #MARL, #simulation-only, #rebalancing |
| Li2019 | WWW 2019 | Synthetic & semi-synthetic ride-dispatch simulator; thousands of agents | mean-field MARL (MF-Q / MF-AC) for order dispatching | Mean-field approximation → tractable thousand-agent dispatch with empirical convergence. | n/a (qualitative) | mean-field assumes interaction symmetry, simulator-only, no fairness | early mean-field MARL formulation for the line | #MARL, #mean-field, #simulation-only |
| Hao2023 (GAT-MF) | KDD 2023 | 100-agent grid-world + two real-data city scenarios with >3,000 agents | graph-attention weighted mean field MARL | Graph-attention-weighted mean field → 42.7% performance gain, 86.4% training-time and 19.2% GPU saving across heterogeneous benchmarks (not all ride-hailing-specific). | +42.7% performance (heterogeneous benchmarks, not ride-hailing dispatch alone); −86.4% training time; −19.2% GPU | "performance" is task-specific and aggregated across non-ride-hailing benchmarks; interaction-strength weighting needs dense computation in dense graphs | extends mean-field with attention; supports the "MF + GAT" hybrid as a strong default | #MARL, #mean-field, #GAT, #real-data, #simulation-only |
| Enders2023 | L4DC 2023 | AMoD simulator (TUM); thousands of vehicles | hybrid MARL: per-agent SAC actor + centralized weighted bipartite matching | Factorized actor + centralized matcher → 5–17% profit gain over central CO-only and joint-action baselines (paper claim, **unverified**). | 5–17% profit (paper claim, **unverified** — v2 does not use this number as theme-anchoring evidence) | simulator-only, profit-only objective, fairness/welfare not modeled, deterministic supply | early hybrid template (MASAC + bipartite matching) replicated by 2024–25 follow-ons | #MARL, #bipartite-matching, #AMoD, #simulation-only |
| Hoppe2024 | L4DC 2024 | AMoD simulator extending Enders2023 | global-reward design + credit assignment for hybrid MADRL | Global vs local reward in hybrid MADRL → 2–6% performance lift attributable specifically to reward globalization (paper claim, **unverified**). | 2–6% performance (paper claim, **unverified** — v2 does not use this number as theme-anchoring evidence) | simulator-only, addresses one design decision in isolation, no fairness | contests local-reward defaults; supports global rewards with credit assignment | #MARL, #global-reward, #AMoD, #simulation-only |
| Wang2025 (CoopRide) | KDD 2025 | Three real-world ride-hailing datasets, millions of orders; grid-as-agent | GNN-based MARL with within- and cross-grid action unification, dynamic cooperation rewards | City-scale all-grid cooperation via GNN-encoded graph state → up to 12.4% over reported baselines. | up to 12.4% over reported baselines | training cost grows with full-graph coupling, real-world deployment (vs. logged eval) not reported, evaluation choice of "SOTA" varies by dataset | current strongest public MARL baseline at city scale | #MARL, #GNN, #real-data |
| Hu2025 (BMG-Q) | T-ITS 2025 (arXiv Jan 2025) | NYC taxi trip data; thousands of vehicles | localized bipartite-match graph attention DDQN + ILP coordinator (GATDDQN) | Local bipartite match graph attention + ILP posterior score → ~10% reward gain, >50% overestimation reduction in ride-pooling. | ~+10% accumulated reward; >50% overestimation reduction | NYC-only, ride-pooling-specific, ILP cost grows with batch size | supports graph-attention DDQN + ILP for ride-pooling; counters overestimation bias of independent DQN | #MARL, #GAT, #bipartite-matching, #ride-pooling, #real-data |
| Jusup2025 | arXiv 2025 (under review) | Continuous-state ride-sourcing simulator; tens of thousands of vehicles | constrained mean-field RL with explicit accessibility/fairness constraints | Constrained MFRL → scales to tens of thousands of vehicles with provable accessibility constraint satisfaction. | n/a (theoretical guarantee + simulator gains) | simulator-only, accessibility metric must be specified ex ante, theoretical bounds depend on regularity assumptions | extends MFRL with hard constraints; supports fairness/accessibility as primary, not post-hoc | #MARL, #mean-field, #constraints, #fairness, #simulation-only |

### 2.3 Thread C — AMoD hybrid learning + optimization (4 papers)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Gammelli2021 | IEEE CDC 2021 | Chengdu and NYC, 4×4 grid (16 stations); Poisson demand | bi-level GNN-RL: A2C with graph convolutional encoder over a network-flow LP inner solver | Bi-level GNN-RL for AMoD rebalancing → 1.6–2.2% deviation from oracle and zero-shot transfer across cities. | 1.6% NYC, 2.2% Chengdu deviation from oracle | small grid, Poisson demand, simulator-only, single-vehicle-class | seminal bi-level GNN-RL+LP architecture (paradigm-defining for the Pavone-lab AMoD line) | #GNN-RL, #bi-level, #LP, #AMoD, #rebalancing, #simulation-only, #cross-city-transfer |
| Gammelli2023 | ICML 2023 | Synthetic min-cost flow; supply-chain (IF2S/3S/10S); DVR with NYC and Shenzhen data | bi-level GNN-RL framework generalized: MPNN/GCN/GAT + linear inner control problem | Bi-level GNN-RL for general network-flow control → 86.7%–99.5% of oracle across heterogeneous network problems. | 86.7%–99.5% of oracle | linear-only inner problem, action space grows with network, large abstract networks not validated | generalizes the AMoD bi-level paradigm to network-flow control beyond mobility | #GNN-RL, #bi-level, #LP, #network-flow, #simulation-only |
| Singhal2024 | ECC 2024 | SF and NYC; 5–20 spatial regions; fleet ≈ 20% of peak demand | bi-level GNN-RL on space-charge-time graph for electric AMoD; SAC | Space-charge-time GNN-RL for electric AMoD → 89% of oracle profit with ~100× LP speedup. | 89% of oracle; 100× speedup | "perfect-foresight demand" baseline assumption, fleet small relative to demand, charging-network configuration fixed | extends GNN-RL to electric AMoD with charging coupling | #GNN-RL, #bi-level, #electric-AMoD, #charging |
| Tresca2026 | TCNS 2026 (in press) | NYC macroscopic 16-station; Luxembourg microscopic + mesoscopic; Brooklyn and Shenzhen real trip data | hierarchical GNN-RL+LP with online/offline/meta-RL variants; multi-fidelity evaluation | Hierarchical GNN-RL+LP scales to robo-taxi fleets → within 2.3% of oracle, 1.5% loss across fidelity, 18% gain via meta-RL on disturbance. | 2.3% from oracle (NY); 1.5% cross-fidelity; +18% via meta-RL | macroscopic↔mesoscopic gap remains, offline RL dataset dependence, reward tuning sensitivity | demonstrates the bi-level paradigm at robo-taxi city scale and across fidelities | #GNN-RL, #bi-level, #robo-taxi, #meta-RL, #cross-city-transfer |

### 2.4 Thread D — pricing (4 papers)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Besbes2021 | Management Science 67(3) 2021 | Theoretical fluid model with strategic spatial supply response | measure-theoretic spatial pricing; equilibrium analysis | Local myopic surge leaves money on the table → globally network-anticipating pricing strictly dominates when supply relocates. | n/a (comparative-static, sign of effect) | continuum-supply abstraction, no congestion or capacity constraints, deterministic | foundational result that local-only surge is suboptimal | #spatial-pricing, #equilibrium, #fluid-model |
| Afeche2023 | MSOM 25(5) 2023 | Theoretical fluid network with strategic drivers and rejection capability | game-theoretic equilibrium analysis with admission control | Strategic-driver model with admission control → in some regimes platform should reject demand to induce repositioning. | n/a (comparative-static; regime-dependent) | strategic-rider behavior simplified, fluid-only, single-class supply | shows demand rejection can be optimal under strategic drivers; broadens platform-control toolkit | #strategic-drivers, #equilibrium, #fluid-model |
| Castillo2025 [v2 corrected] | Econometrica 93 2025 | Uber data, Houston market | structural empirical model of surge incidence | **Surge pricing increases** total welfare by 3.53% relative to a uniform-pricing counterfactual; rider surplus +6.97% with surge; driver surplus −1.97% with surge. | **+3.53% total welfare with surge; +6.97% rider surplus with surge; −1.97% driver surplus with surge** (v2 corrected — v1 had inverted the direction, reading these as the effects of *eliminating* surge) | one-city study (Houston only), identification depends on reduced-form variation, model abstracts from rider learning, generalization to other markets unverified | settles a canonical question on surge welfare: aggregate welfare gain comes with redistribution from drivers to riders | #surge-pricing, #welfare, #structural-empirical, #real-data |
| Zhang2023 | AAAI 2023 | NYC taxi network simulator | future-aware joint pricing-matching ADP for ride-pooling | Joint future-aware pricing+matching → +17% revenue, −14% vehicle requirement (paper claim, **unverified**). | +17% revenue, −14% vehicles (paper claim, **unverified** — v2 does not use this number as theme-anchoring evidence) | simulator-only, single-pool size, no equilibrium agents, ADP scaling | flagship SMU joint pricing+matching DRL contribution; bridges D and E threads | #joint-pricing-matching, #ADP, #ride-pooling, #simulation-only |

### 2.5 Thread E — LLMs, foundation models, and joint optimization (4 LLM/FM papers; Sun2024 cross-referenced from Thread B)

The v1 draft placed Sun2024 (JDRCL) inside Thread E "for joint-optimization comparability." The peer review correctly noted that this creates a categorical inconsistency — JDRCL is a constrained MARL paper, not an LLM/foundation-model paper. v2 places JDRCL in Thread B (MARL, with the constrained-MFRL/Jusup2025 family) for the schema and matrices, and **cross-references it from Thread E only when the comparison axis is "joint matching+repositioning"** (Matrix 5 row 5; Theme 5).

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Yuan2024 (UniST) | KDD 2024 | 20+ urban ST datasets across 15 cities (taxi/bike/traffic) | ST foundation model with masked pretraining + knowledge-guided prompts | Pretraining + prompt-tuning on heterogeneous ST data → strong zero/few-shot transfer; 10.1% over no-pretrain baseline. | MAE 26.84 (zero-shot); +10.1% over no-pretrain | not a controller (forecasting only), prompt design heuristic, semantic understanding remains shallow | supports ST foundation model paradigm for cross-city/zero-shot prediction | #foundation-model, #ST-foundation-model, #zero-shot, #cross-city-transfer |
| Li2024 (UrbanGPT) | KDD 2024 | NYC taxi/bike/crime ST datasets; zero-shot prediction | LLM as encoder via spatio-temporal dependency encoder + instruction tuning | Instruction-tuned ST-LLM encoder → zero-shot ST prediction; ~28% over generalization baselines. | MAE 6.16 NYC-taxi zero-shot; ~+28% over generalization baselines | LLM inference cost, zero-shot only, not a controller | supports LLM-as-encoder for cross-domain ST prediction | #LLM, #ST-foundation-model, #zero-shot, #cross-city-transfer, #instruction-tuning |
| Han2025 (GARLIC) | AAAI 2025 | Manhattan taxi (4h) and Hangzhou ride-hailing (30d); two real-world datasets | GPT-augmented RL: multiview-graph state, hierarchical traffic state, GPT-driven dynamic reward learning (contrastive) | GPT-driven reward shaping over hierarchical traffic state → −7.6 pp empty-loaded rate vs offline RL. | empty-loaded rate 38.23% (vs. 40.71% best baseline); 0.30 km error | LLM compute heavy, V2X latency 50–200 ms, single-city evaluation per dataset | flagship peer-reviewed LLM+RL ride-hailing paper; uses LLM as reward designer, not controller | #LLM, #GPT, #joint-matching-repositioning, #MARL, #real-data |
| Lyu2025 (LLM-ODDR) | arXiv preprint 2025 | Manhattan, Chengdu, NYC taxi/ride-hailing data; small-to-large fleet sizes | LLM-as-agent for joint order dispatching + driver repositioning; multi-objective refinement; fine-tuned JointDR-GPT | LLM-as-agent for joint matching+repositioning → +5.21–48.87% GMV vs traditional RL across fleet sizes. | +5.21% (small fleet) to +48.87% (large fleet) GMV (paper claim) | inference latency 1–5 s/decision, interpretability, dependence on fine-tuning data, no peer review yet | first systematic LLM-as-agent for joint matching+repositioning; supports LLM as planner | #LLM, #joint-matching-repositioning, #fairness, #real-data |

**Cross-referenced from Thread B for Thread E joint-optimization comparability:**

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Sun2024 (JDRCL) | TKDE 2024 (KDD'22 ext.) | Haikou, Chengdu, NYC datasets | constrained multi-objective Markov game; max-min fairness; joint dispatch+repositioning MARL | Constrained max-min fairness MARL → joint matching+repositioning with convergence guarantee. | qualitative GMV vs KM gap (table-level); fairness entropy materially lifted | single-agent action-dim explosion, slow convergence, fairness vs efficiency Pareto not characterized analytically | sole peer-reviewed beachhead for joint matching+repositioning with first-class fairness | #joint-matching-repositioning, #fairness, #max-min-fairness, #MARL, #real-data |

### 2.6 Tag dictionary

The corpus uses the following controlled vocabulary. The v1 draft claimed "18 reconciled tags" while listing 50+ distinct strings; v2 corrects the count and explicitly groups the tags by axis. Total: **52 distinct tags across six axes.** Reconciliation merged near-duplicates (e.g., `#joint-three-way` is reserved for matching+pricing+rebalancing simultaneously and appears on no paper in the corpus, since no paper fully satisfies it).

- **Method axis (15):** `#online-RL`, `#offline-RL`, `#end-to-end`, `#value-function`, `#bipartite-matching`, `#MARL`, `#mean-field`, `#GNN-RL`, `#GAT`, `#GNN`, `#bi-level`, `#LP`, `#network-flow`, `#ADP`, `#ensemble`.
- **LLM/FM axis (5):** `#LLM`, `#GPT`, `#foundation-model`, `#ST-foundation-model`, `#instruction-tuning`, `#prompt-tuning`. (Six listed; `#prompt-tuning` and `#instruction-tuning` are kept distinct per UniST/UrbanGPT methodology.)
- **Application/operational axis (10):** `#matching`, `#rebalancing`, `#electric-AMoD`, `#charging`, `#robo-taxi`, `#AMoD`, `#ride-pooling`, `#joint-pricing-matching`, `#joint-matching-repositioning`, `#joint-three-way`.
- **Pricing/economics axis (8):** `#strategic-drivers`, `#equilibrium`, `#fluid-model`, `#structural-empirical`, `#welfare`, `#spatial-pricing`, `#surge-pricing`, `#meta-RL`.
- **Constraint/objective axis (3):** `#fairness`, `#max-min-fairness`, `#constraints`.
- **Data/deployment axis (8):** `#real-data`, `#simulation-only`, `#production-deployment`, `#didi`, `#lyft`, `#safety-exploration`, `#cross-city-transfer`, `#zero-shot`, `#global-reward`. (Nine listed; `#global-reward` is method-axis-adjacent but is included here because it appears as a deployment/design choice in Hoppe 2024.)

(The duplication between axes for borderline tags like `#meta-RL` and `#global-reward` is intentional and noted; v3 should re-reconcile to a strictly partitioned dictionary.)

---

## 3. Synthesis matrices (Step 3 product)

The matrices below take a *lateral* view of the corpus along five comparison axes. Each matrix has a single axis of comparison, as the guide requires. Empty cells are marked `unknown` rather than fabricated. Where two papers do not directly compare on a given axis, a `comparability_notes` row is added.

### 3.1 Matrix 1 — matching/dispatching: data regime × action representation

*Comparison axis*: **how is the matching policy learned and what does the policy output?**

| paper | data regime | output representation | combinatorial step | deployed? | headline number | comparability_notes |
|---|---|---|---|---|---|---|
| Xu2018 | offline-trained value table, online use | value(state) → edge weight | Hungarian | yes (Didi) | 0.5–5% revenue | reference baseline |
| Eshkevari2022 | online state-value iteration | value(state) → edge weight | bipartite matching with smoothing & MAB pruning | yes (Didi int'l) | 1.3% / up to 5.3% income | online learning + safeguards |
| Yang2024 | offline RL on logged data | value(state) of region as future-impact term | KM (default); GS (alternative) | pilot only | ~5–6% IORR/IGMV (City B OPE) [v2 corrected] | replaces region-value independence assumption |
| Yue2024 | offline RL on Didi simulator + data | end-to-end o–d assignment | none (subsumed) | unspecified beyond simulator | 0.7–3.9% TDI / 1–2% CR | single-stage replacement |
| Zhang2024 (NondBREM) | offline RL only | discrete action → assignment | implicit | no (offline eval) | 3.76% response rate | safety-first formulation |
| Azagirre2024 (Lyft) | online RL | future-earnings estimator → matching scores | bipartite matching | yes (Lyft global) | $30M/yr | non-Didi production |
| Wang2025 (CoopRide) | offline + simulator | grid-level cooperation actions; dispatch within & cross grid | implicit per-grid | unspecified | up to 12.4% | grid-as-agent, all-grid coop |
| Hu2025 (BMG-Q) | simulator + NYC trips | per-agent Q over local bipartite match graph | ILP with posterior score | no | ~+10% reward, >50% overest. ↓ | ride-pooling specific |
| Enders2023 | simulator | per-agent SAC continuous embedding → matching weights | weighted bipartite matching | no | 5–17% (**unverified**) | hybrid template ablation; not theme-anchoring in v2 |
| Hoppe2024 | simulator | reward-shaped per-agent embedding | bipartite matching | no | 2–6% (**unverified**) | reward-design ablation; not theme-anchoring in v2 |
| Lin2018 | simulator | contextual A2C / DQN per cell | implicit | no | n/a | early MARL fleet baseline |
| Li2019 | simulator | mean-field Q over neighbors | implicit | no | n/a | early MFMARL baseline |
| Hao2023 (GAT-MF) | simulator + city data | weighted-MF Q with attention | implicit | no | +42.7% perf (heterogeneous benchmarks), −86.4% time, −19.2% GPU | scales to >3000 agents |
| Jusup2025 | continuous-state simulator | mean-field policy with Lagrangian constraints | implicit | no | n/a | constraint-feasible at 10⁴ vehicles |
| Han2025 (GARLIC) | simulator + 2 city datasets | RL policy with GPT-shaped reward | bipartite matching | no | empty-load rate −7.6 pp | LLM as reward designer |
| Lyu2025 (LLM-ODDR) | three city datasets (varied fleet) | LLM directly proposes assignment + reposition | implicit | no | +5.21–48.87% GMV (paper claim) | LLM as planner |
| Sun2024 (JDRCL) | three city datasets | per-driver MARL policy under constraint | implicit | no | qualitative GMV gap; fairness lift | constrained joint matching+reposition |

*Comparability notes for Matrix 1.* Two-stage pipelines (Xu2018, Eshkevari2022, Azagirre2024, Yang2024, Wang2025) and end-to-end pipelines (Yue2024, Lyu2025) are rendered comparable only on the deployment-readiness axis; their headline numbers measure different objectives (revenue, response rate, GMV, TDI), so direct numeric comparison is not warranted. The "headline number" column is used for direction-of-effect, not magnitude comparison. Numbers flagged **unverified** are not used to anchor synthesis claims.

### 3.2 Matrix 2 — MARL coordination geometry: agent definition × cooperation scope

*Comparison axis*: **what is an "agent" and how far does cooperation extend?**

| paper | agent definition | cooperation scope | mechanism | scaling demonstrated |
|---|---|---|---|---|
| Lin2018 | grid cell | cell-local context | shared global state vector | NYC simulator |
| Li2019 | driver | mean-field (all neighbors symmetric) | mean-field Q | thousands |
| Hao2023 (GAT-MF) | agent (driver-like) | weighted mean field with attention | attention weights over neighbors | >3000 |
| Enders2023 | vehicle | central matcher, decentralized actor | MASAC + bipartite matching | hundreds–thousands |
| Hoppe2024 | vehicle | central matcher, reward-globalized | global reward + credit assignment | hundreds–thousands |
| Jusup2025 | vehicle | mean-field with constraints | constrained MFRL | 10⁴ |
| Wang2025 (CoopRide) | grid | all-grid cooperation | GNN encoded + within/cross-grid action unification | city-scale |
| Yang2024 (GRC) | region | cooperative Markov game with goal-state credit | environment model + scoring + policy | city pilots |
| Sun2024 (JDRCL) | driver | constrained multi-objective game | max-min fairness + MARL | three-city scale |
| Hu2025 (BMG-Q) | vehicle (in pool) | local bipartite-match graph | GAT + ILP | thousands (NYC) |
| Lyu2025 (LLM-ODDR) | LLM-controlled vehicle | LLM-mediated coordination | prompt-driven joint reasoning | fleet sizes 100–500 |

*Comparability notes for Matrix 2.* The matrix exposes a lateral pattern: papers either define the agent as a **driver/vehicle** (and rely on mean-field or attention-weighted mean-field to escape the joint action space) or as a **grid/region** (and treat cooperation as inter-grid information flow). Hybrid forms exist (Wang2025 uses grid agents but adds within-grid + cross-grid action types to recover driver-level expressivity). **No paper in the corpus uses the rider as an agent**, despite riders being strategic in the OR/economics literature (Castillo 2025 estimates rider price-elasticity from data; Besbes 2021 carries strategic riders as a structural feature). This is the single most striking absence in the matrix and is taken up in Section 6.

### 3.3 Matrix 3 — bi-level GNN-RL + optimization paradigm: extension axis

*Comparison axis*: **what does each Pavone-lab and TUM bi-level paper extend?**

| paper | extension axis | inner problem | empirical finding |
|---|---|---|---|
| Gammelli2021 | scope: introduce bi-level GNN-RL+LP for AMoD rebalancing | LP rebalancing | within 1.6–2.2% of oracle; cross-city zero-shot |
| Gammelli2023 | scope: generalize to abstract network-flow control | LP/linear inner | 86.7%–99.5% of oracle across DVR, supply chain, min-cost flow |
| Singhal2024 | physics: incorporate charging and power | LP on space-charge-time graph | 89% of oracle, 100× speedup |
| Tresca2026 | scale + transfer: macroscopic→mesoscopic; meta-RL across cities | hierarchical LP | 2.3% from oracle; 1.5% cross-fidelity; +18% meta-RL gain |
| Enders2023 | architecture: bring bi-level idea to MARL via MASAC + bipartite matching | bipartite matching | 5–17% profit (**unverified**) |
| Hoppe2024 | reward: study global vs local rewards within Enders' bi-level | bipartite matching | 2–6% (**unverified**) |

*Comparability notes for Matrix 3.* The Pavone-lab progression (Gammelli2021 → Gammelli2023 → Singhal2024 → Tresca2026) extends along three orthogonal axes: scope, physics, and scale + transfer. The TUM line (Enders2023, Hoppe2024) extends along an architectural axis — replacing the LP inner problem with bipartite matching and per-agent SAC. Both lines accept the bi-level template; the disagreement is on whether the inner problem should be a network-flow LP or a bipartite-matching CO problem. The TUM line's specific magnitude claims are unverified in v2 and are reported here as direction-of-effect only.

### 3.4 Matrix 4 — pricing: methodological tradition × strategic-agent assumption

*Comparison axis*: **what tradition is the pricing analysis in, and what is assumed about strategic agents?**

| paper | tradition | strategic riders? | strategic drivers? | what is solved | numeric headline |
|---|---|---|---|---|---|
| Besbes2021 | OR / equilibrium fluid | partial | yes (relocate) | optimality of network-anticipating pricing | comparative-static (sign) |
| Afeche2023 | OR / equilibrium fluid | partial | yes (reject demand) | regime conditions for admission control | comparative-static (regime-dependent) |
| Castillo2025 [v2 corrected] | structural empirical | yes (price-elastic demand) | yes (entry, hours) | welfare incidence of surge | **+3.53% total welfare with surge; +6.97% rider surplus with surge; −1.97% driver surplus with surge** |
| Zhang2023 | DRL / ADP | no | no | joint future-aware pricing+matching for pooling | +17% revenue, −14% vehicles (**unverified**) |

*Comparability notes for Matrix 4.* The matrix exposes that the pricing literature in this corpus is split along an **assumption** axis: OR/economics papers carry strategic agents as a structural feature, while the single DRL pricing paper in the corpus does not. **However, this controversy claim is overstated when the bounded-out MFG-RL literature is taken into account** — see Section 11 and Theme 4 (Section 5.4) for the corrected framing. Castillo 2025's estimates are scoped to a single market (Houston) and are not validated cross-market; the welfare-incidence claim should be read as one estimate from one structural model on one platform.

### 3.5 Matrix 5 — LLM/foundation-model role × ride-hailing operational role

*Comparison axis*: **what does the LLM/foundation model actually do in the pipeline?**

| paper | role | controller? | benchmark | output |
|---|---|---|---|---|
| Yuan2024 (UniST) | forecaster (encoder + prompt) | no | 20+ ST datasets, 15 cities | spatiotemporal predictions |
| Li2024 (UrbanGPT) | encoder for forecasting (instruction-tuned) | no | NYC taxi/bike/crime | spatiotemporal predictions |
| Han2025 (GARLIC) | reward designer (GPT shapes reward) | no (RL still acts) | Manhattan, Hangzhou | shaped reward → dispatching policy |
| Lyu2025 (LLM-ODDR) | direct planner / controller | yes | Manhattan, Chengdu, NYC | direct dispatch + reposition decisions |
| Sun2024 (JDRCL) [cross-ref from Thread B] | n/a (no LLM; cross-referenced for joint-optimization comparability only) | n/a | Haikou, Chengdu, NYC | joint matching+reposition under fairness constraint |

*Comparability notes for Matrix 5.* No paper in the corpus uses an LLM/foundation model as a **simulator** (although the deep-research note flags this as an open direction) or as a **pricing engine** (also unaddressed). The only paper that uses the LLM as a direct controller is `Lyu2025`, which is a preprint; the only peer-reviewed LLM ride-hailing paper (Han2025) uses the LLM as a reward designer. The role-taxonomy of "what does the LLM do" is therefore not yet standardized in the field. Sun2024 is **cross-referenced from Thread B** rather than placed in Thread E, since it is not an LLM/FM paper; including its row here is purely for joint-optimization side-by-side reading.

---

## 4. Field overview (background paragraph)

The 2018–2026 ride-hailing ML literature in this corpus has converged on **deep RL with GNN encoders and bipartite-matching combinatorial solvers** as the dominant matching/rebalancing stack, with a fast-growing **LLM/foundation-model** layer on top and a parallel **OR/economics pricing equilibrium** track that has only partially intersected with the ML side. The empirical center of gravity in dispatching shifted from single-agent value learning (Xu2018) to multi-agent RL with mean-field approximations (Li2019, Hao2023) and then to graph-attention MARL with grid-as-agent or vehicle-as-agent variants (Wang2025, Hu2025). In autonomous mobility-on-demand the bi-level GNN-RL + linear-program template (Gammelli2021, Gammelli2023, Singhal2024, Tresca2026) is dominant; in deployed dispatch, the production lineage at Didi (Xu2018, Eshkevari2022, Yang2024, Yue2024) and Lyft (Azagirre2024) is the strongest empirical base. On the pricing side, the OR/economics line has largely settled the foundational welfare and spatial-pricing questions (Besbes2021, Afeche2023, Castillo2025), while DRL-based joint pricing and matching is rare in this corpus and not yet in dialogue with the equilibrium results (Zhang2023) — though, as Section 11 acknowledges, the bounded-out MFG-RL literature is exactly the bridge. The LLM thread is conceptually the broadest but empirically the thinnest: foundation models for urban ST prediction (Yuan2024, Li2024) and one peer-reviewed LLM-as-reward-designer (Han2025) coexist with an LLM-as-controller preprint (Lyu2025) and one peer-reviewed joint matching+repositioning under fairness constraints (Sun2024).

---

## 5. Themes (Step 4 product) and MEAL paragraphs (Step 6 product)

This section presents six themes drawn from the matrices: two consensus, two controversy, two gap. Each theme is one MEAL paragraph (Main idea, Evidence, Analysis, Link), written so that the grammatical subjects of sentences are concepts and findings, not author names. Every theme cites the specific rows in the matrices that support and (where applicable) contradict it. Numbers flagged `(unverified)` in Section 2 are not used as theme-anchoring evidence; they appear, where they appear at all, only as illustrative auxiliary references.

### 5.1 Theme 1 (consensus, with active contestation) — RL-proposed weights plus a combinatorial matcher is the dominant matching architecture in this corpus, but it is no longer uncontested

**Main idea.** Within the corpus reviewed, the most common architectural template for matching is "RL learns edge weights on the driver–order bipartite graph; a combinatorial step (Hungarian, ILP, or Gale–Shapley) takes those weights and produces a feasible assignment" — but two recent papers (Yue2024, Lyu2025) explicitly contest the template, and a thesis-style review should not call the template "won." **Evidence.** This template appears in every deployed system in the corpus — Didi's learning + planning approach (Xu2018), Didi's online state-value iteration (Eshkevari2022), Lyft's online matching RL (Azagirre2024) — as well as in the strongest non-deployed graph-MARL baselines (Wang2025; Hu2025, where the combinatorial step is an ILP with a learned posterior score). The bi-level AMoD line (Gammelli2021, Gammelli2023, Singhal2024, Tresca2026) follows the same template with an LP rather than a bipartite-matching CO solver. Two papers explicitly contest the template: Yue2024 ("D2SN") replaces the two-stage pipeline with a single end-to-end DRL model and reports verified 0.7–3.9% TDI gains on Didi data, and Lyu2025 has the LLM directly emit assignment+reposition actions with paper-claimed gains across fleet sizes (preprint, not peer reviewed). The TUM hybrid MARL line (Enders 2023, Hoppe 2024) is consistent with the template but its specific magnitudes are unverified. **Analysis.** The persistence of the two-stage template is best explained by feasibility: the matching step has hard constraints (one-driver-per-order, geo-fenced batches) that the RL action space cannot easily encode, while the combinatorial solver guarantees feasibility regardless of how noisy or out-of-distribution the learned weights are. End-to-end approaches inherit the burden of feasibility through architectural tricks (autoregressive decoding, hold-actions). **Link.** A current research project in this space should default to the two-stage template unless it has a specific architectural reason to depart from it; if it does depart, it should report the same matching feasibility guarantees the two-stage template provides for free, and benchmark explicitly against Yue2024's verified 0.7–3.9% TDI as the bar to beat.

### 5.2 Theme 2 (consensus within the corpus reviewed) — graph encoders are the standard state representation for spatial dependence

**Main idea.** Within the corpus reviewed, the spatial state of a ride-hailing system is routinely encoded with graph neural networks (GNNs), and the choice of GNN variant — GCN, GAT, MPNN — is treated as a tunable rather than a load-bearing modeling decision. The corpus is not unbiased (the bounded-out ST-GNN forecasting literature is excluded), so the claim is scoped to "within the corpus" rather than to the field as a whole. **Evidence.** The bi-level AMoD line uses graph convolutions (Gammelli2021), and its 2023 generalization explicitly studies MPNN/GCN/GAT variants and reports that graph-based encoders dominate MLP/CNN baselines via permutation invariance and locality (Gammelli2023). Graph attention is the most common encoder in current MARL: GAT-MF combines attention with mean field for >3,000 agents (Hao2023, with the caveat that the 42.7% headline is across heterogeneous benchmarks not all of which are ride-hailing); BMG-Q uses graph-attention DDQN over a localized bipartite match graph (Hu2025); CoopRide encodes city-scale grid relationships with a GNN (Wang2025). UrbanGPT and UniST treat the spatiotemporal graph as the substrate for foundation-model pretraining (Yuan2024, Li2024). The only papers in the corpus that do not use a GNN are the OR/economics pricing papers, which work in continuous space (Besbes2021, Afeche2023, Castillo2025), and the pre-GNN-era MARL formulations (Lin2018, Li2019). **Analysis.** Two structural facts make GNNs an attractive default: cities are explicitly graph-structured (zones, road segments, charging stations) and ride-hailing decisions exhibit spatial locality at the right scale for message-passing depth of 1–3 hops. The remaining methodological choice — attention vs. fixed-weight aggregation — appears to matter less than whether some graph encoder is present at all, but this conclusion is sensitive to the corpus selection and may not survive expansion to the ST-GNN forecasting literature where attention vs. convolution debates are active. **Link.** A new ride-hailing controller can adopt graph attention (GAT) or message-passing (MPNN) as the default encoder without a methodological justification beyond "this is the corpus standard"; effort is better spent on the *coordination geometry* (Theme 3) than on the encoder choice.

### 5.3 Theme 3 (controversy) — agent definition: driver/vehicle vs. grid/region

**Main idea.** The corpus is split on whether the basic MARL agent should be a **driver/vehicle** or a **grid/region**, and the choice has cascading consequences for credit assignment, action space size, and how cooperation is modeled. **Evidence.** Driver-as-agent appears in early mean-field MARL (Li2019), the Lyft and Didi production deployments (Azagirre2024, Eshkevari2022), the TUM hybrid MARL line (Enders2023, Hoppe2024), the BMG-Q ride-pooling line (Hu2025), the constrained MFRL line (Jusup2025), and the LLM controllers (Lyu2025). Grid-as-agent appears in CoopRide (Wang2025), GAT-MF's grid-world variant (Hao2023), Yang2024's region cooperative game, and the original contextual MARL (Lin2018). Production deployments lean driver-as-agent, while the strongest 2024–25 academic MARL benchmarks (Wang2025) use grid-as-agent. **Analysis.** The driver-as-agent formulation is closer to the operational reality (drivers are the decision-makers) and inherits cleaner credit assignment in the small-fleet regime, but the joint-action space grows with fleet size and forces mean-field or attention-weighted approximations. The grid-as-agent formulation collapses the action space (one action per cell) but loses driver-level personalization and creates a new credit-assignment problem because grid-level rewards do not directly attribute to driver-level decisions. Yang2024 makes the credit-assignment problem of grid-as-agent explicit (the FOVIR formulation is exactly the question of how a region's value affects others) and proposes goal-reaching collaboration as the answer. CoopRide partially closes the gap by giving each grid both within-grid and cross-grid actions, recovering some driver-level expressivity. **Link.** A research project should pick the agent definition by what its evaluation metric looks like: revenue/GMV-style metrics favor grid-as-agent (cleaner aggregation), while individual-driver metrics (fairness, income variance, acceptance rate) favor driver-as-agent.

### 5.4 Theme 4 (controversy, with corrected framing) — DRL pricing within this corpus does not engage with OR/economics pricing equilibrium results, but the bounded-out MFG-RL literature is exactly the bridge

**Main idea.** Within the corpus reviewed, the pricing thread contains two largely disconnected literatures: an OR/economics equilibrium track that has settled foundational welfare and spatial-pricing questions, and a single DRL pricing paper (Zhang2023) that does not engage with those equilibrium results. **The v1 framing of this controversy as a global feature of the field is overstated**, however: the mean-field-game RL literature (bounded out of v2; see Section 11) is precisely the bridge between OR equilibrium pricing and DRL control, and the controversy here should be read as a *coverage gap in this corpus* rather than as a global disconnect. **Evidence.** The OR/economics track has shown that local myopic surge is suboptimal under strategic supply (Besbes2021), that admission control can be optimal under strategic drivers (Afeche2023), and — with the v2 correction — that surge pricing **increases** total welfare by 3.53% relative to a uniform-pricing counterfactual on Uber Houston data, with riders gaining 6.97% in surplus and drivers losing 1.97% (Castillo2025). **The Castillo result is single-market (Houston) and the cross-market generalization is unverified.** The DRL pricing line is represented in this corpus by joint future-aware pricing + matching for ride-pooling (Zhang2023), which reports +17% revenue and −14% vehicle requirement on a NYC simulator (paper-claimed; unverified) but does not model strategic drivers, does not characterize equilibrium, and does not benchmark against the welfare-incidence findings. **Analysis.** The disconnect within this corpus is partly a venue mismatch (Econometrica/MS vs. AAAI/KDD) and partly a methodological mismatch (closed-form fluid analysis vs. simulator-based DRL). A deeper structural reason is that DRL pricing in this corpus optimizes platform revenue while the OR/economics track has shown that revenue and welfare can diverge under heterogeneous distributional incidence. Without modeling strategic agents and without engaging the equilibrium-aware MFG-RL literature, simulator-based DRL pricing risks recommending policies whose welfare implications the equilibrium literature has already characterized. **Link.** A new pricing project should adopt the strategic-driver assumption from the OR/economics literature as a baseline; it should benchmark its DRL policy against the comparative-static predictions in Besbes2021 and the welfare decomposition in Castillo2025 (with explicit Houston/Uber scoping), and it should engage with the MFG-RL bridge literature rather than treating ML and OR as disconnected. The reverse direction — DRL informing equilibrium analysis with high-fidelity demand forecasters — is also open.

### 5.5 Theme 5 (gap) — true three-way joint optimization of matching + pricing + rebalancing

**Main idea.** Joint optimization is the visibly emerging frontier, but **no paper in the corpus addresses all three operational levers (matching + pricing + rebalancing) simultaneously with peer-reviewed evidence**. **Evidence.** Pairwise joint optimization exists: matching + repositioning under fairness (Sun2024 JDRCL); matching + repositioning by LLM agent (Lyu2025); pricing + matching for ride-pooling via ADP (Zhang2023, with the unverified-magnitude caveat). The Pavone-lab AMoD line couples matching and rebalancing through bi-level optimization (Gammelli2021, Singhal2024, Tresca2026) but treats pricing as exogenous. The MARL lines (Wang2025, Hao2023, Hu2025) treat pricing as exogenous as well. **Analysis.** Two reasons explain the gap. First, the matching/rebalancing levers act on second-to-minute timescales while pricing acts on minute-to-longer timescales; coupling them in a single Markov decision process inflates the state and action spaces and makes off-policy learning unstable. Second, pricing changes demand, which in turn changes the post-matching idle distribution that rebalancing acts on; a three-way joint policy must internalize that feedback, which the equilibrium literature handles via fluid models but DRL has only partially absorbed. JDRCL's max-min fairness convergence guarantee shows that constrained multi-objective MGs can be analyzed in the pairwise case; extending such analysis to three-way is the unsolved technical step. **Link.** A research project that aims to fill this gap should (a) work in the pairwise matching+rebalancing setting first, with a max-min fairness or similar constraint, before adding pricing; (b) borrow the strategic-driver fluid abstraction from Besbes2021/Afeche2023 as the third decision module; and (c) evaluate the joint policy against the welfare incidence of surge documented in Castillo2025 (with corrected direction) to test whether three-way joint optimization changes the distributional answer.

### 5.6 Theme 6 (gap) — LLM role in ride-hailing is unstandardized and underbenchmarked

**Main idea.** The LLM/foundation-model thread is the most rapidly growing line in the corpus, but the **role** an LLM plays in the pipeline (encoder, forecaster, reward designer, planner, simulator) is not yet standardized, no shared benchmark exists, and only one peer-reviewed LLM-as-controller paper appears in the corpus. **Evidence.** Within the corpus, the LLM is a forecaster in UrbanGPT and UniST (Yuan2024, Li2024), a reward designer in GARLIC (Han2025), and a direct planner in LLM-ODDR (Lyu2025). The first two are pretrained ST foundation models with peer-reviewed cross-city zero/few-shot generalization (UniST 10.1% over no-pretrain; UrbanGPT ~28% over generalization baselines). The third is peer-reviewed at AAAI but uses GPT only to shape rewards, not to act. The fourth (Lyu2025) is the only paper that gives the LLM the controller role and reports +5.21–48.87% GMV across fleet sizes, but it is an unreviewed preprint with 1–5 s/decision inference latency. **Analysis.** The role mismatch is not surprising for an emerging line, but it has two practical consequences. First, claims about "LLM for ride-hailing" cannot be compared head-to-head because the LLMs are in different positions in the pipeline; a 28% gain as encoder is not commensurable with a 7.6 pp empty-load reduction as reward designer. Second, no benchmark in the corpus standardizes city, fleet size, and metric the way the AMoD line did with the macroscopic 16-station simulator (Gammelli2021); the LLM line is therefore at risk of pattern over-fitting on case studies. **Link.** A new project should pick a single LLM role (planner, reward designer, or forecaster) and benchmark against a non-LLM baseline of the same operational class; if the project uses an LLM as a controller, latency and grounding (hallucinated assignments) should be measured explicitly. A useful contribution to the field would be a shared LLM-for-ride-hailing benchmark that fixes city/fleet/metric and varies role.

---

## 6. Where the synthesis matrix exposed surprises (corpus-specific)

Three patterns surfaced in the matrices that were not visible in the per-paper schema entries. The v1 draft listed a fourth "deployed-vs-academic effect-size gap" surprise; the peer review correctly noted that this gap is well-documented in the broader RL literature and is not specific to this corpus, so it has been removed in v2 and replaced with a corpus-specific observation.

A first surprise is that **no paper in the corpus uses the rider as the agent**, despite riders being treated as strategic in the OR/economics track (Castillo 2025 estimates rider price-elasticity from data; Besbes 2021 carries strategic riders as a structural feature). The MARL formulations universally agentize either drivers or grids, and the LLM controller (Lyu2025) agentizes the LLM-controlled vehicle. This is a corpus-specific absence that is invisible in the per-paper schema — every paper has a defensible agent definition for its purpose — but visible only in Matrix 2's lateral view. The absence is consequential for fairness and welfare framings: rider-side metrics (waiting time, surge incidence, allocation equity) are evaluated post-hoc rather than directly optimized.

A second surprise is that **the bi-level paradigm is so dominant in AMoD that the controversy is no longer "RL vs OR" but "which inner problem"**. Gammelli's lab uses an LP for network-flow rebalancing; the TUM lab uses bipartite matching; both share the outer GNN-RL. The methodological agreement is much stronger than a non-specialist would guess from the venue diversity (CDC, ICML, ECC, TCNS, L4DC). Whether this convergence survives the addition of MFG-RL to the corpus (Section 11) is an open question.

A third surprise is that **fairness has only just entered the corpus as a first-class objective**. JDRCL (Sun2024) gives fairness a max-min formulation with a convergence guarantee; Jusup2025 gives accessibility the role of a hard constraint; Lyu2025 includes a fairness-aware dispatching subroutine. Outside these three, fairness in ride-hailing ML is post-hoc or absent. This is a recognizable pattern in ML subfields where a constraint becomes first-class only after the core control problem stabilizes; the literature is at the start of that transition.

---

## 7. Gap analysis (Step 5 product)

The three gap themes (5.5 Three-way joint optimization, 5.6 LLM role standardization, plus a third on cross-city transfer / sim-to-real) are evaluated below against the importance/reasonableness/feasibility criteria. v1 cleared all three uniformly; v2, responding to peer review, attaches at least one *counter-reason* to each gap and lets the verdict turn on which side of the argument is stronger.

### 7.1 Gap A — three-way joint optimization with strategic-agent guarantees

*Importance.* Filling this gap meaningfully changes the answer to the core platform-control question, because the equilibrium literature has already shown that pricing and rebalancing are coupled through strategic supply (Besbes2021, Afeche2023). A controller that optimizes only matching+rebalancing while taking pricing as exogenous misses the equilibrium feedback that determines welfare incidence (Castillo2025, with corrected direction). The importance is therefore a structural property of the problem, not a "no one has done it" framing.

*Reasonableness.* The gap connects to two questions the field cares about: Castillo2025's distributional result shows that surge has heterogeneous welfare incidence on Uber Houston data, and DRL pricing in this corpus has not engaged with that result; JDRCL's convergence guarantee for two-way joint matching+repositioning under max-min fairness shows that three-way extension is a recognizable technical question in the same lineage.

*Feasibility.* Two-way joint matching+repositioning has been done (JDRCL, Lyu2025); three-way joint matching+repositioning+pricing requires either (a) coupling a fluid-model pricing module with a DRL matching+repositioning module, or (b) a single MARL formulation with prices as agent actions. (a) is feasible with current methods; (b) is harder but not impossible — the action space inflation is manageable if pricing acts on a coarser timescale.

*Counter-reason* (added in v2). The OR/economics fluid-model line has, in fact, solved versions of three-way joint optimization analytically (Besbes 2021's network-anticipating pricing implicitly couples spatial pricing with the supply repositioning it induces; Afeche 2023 couples admission control with relocation incentives). The right framing of Gap A may therefore be "the ML literature should adopt the equilibrium framing rather than reinvent three-way coupling in MDPs," not "three-way joint optimization has not been done." Under that framing, the gap is partly absorbed by Gap B-style integration work (bridging MFG-RL and OR equilibria) rather than being a standalone unsolved problem.

*Verdict (v2).* Real gap on the *implementation-as-DRL* axis (no peer-reviewed three-way DRL paper exists in this corpus); partly closed on the *analytical solvability* axis (OR equilibrium analysis has solved adjacent versions). A research project taking up this gap should be explicit about which axis it is targeting and should not claim novelty against the OR-equilibrium prior art.

### 7.2 Gap B — LLM role taxonomy and shared benchmark

*Importance.* The LLM line is growing fast and the role mismatch (encoder, forecaster, reward designer, planner) makes head-to-head comparison impossible. A standardized benchmark is therefore high-importance for any near-term consolidation of the line.

*Reasonableness.* The AMoD subfield's bi-level GNN-RL line consolidated rapidly once Gammelli2021's macroscopic 16-station benchmark appeared; the LLM line is at the same pre-benchmark stage and would benefit from a similar unifying benchmark. The deep-research note flags this as an open frontier explicitly.

*Feasibility.* A benchmark requires (a) a shared simulator/data — most LLM ride-hailing papers already use Manhattan or NYC taxi data, so a NYC-taxi standard is plausible; (b) a shared metric — empty-load rate or GMV, both already widely reported; (c) a shared role-axis — encoder/forecaster/reward-designer/planner. Building a benchmark that fixes (a) and (b) and varies (c) is feasible within a single research project.

*Counter-reason* (added in v2). The LLM line in this corpus is only four papers (Yuan2024, Li2024, Han2025, Lyu2025); a "shared benchmark" with such a small N risks overfitting the benchmark to the four extant role choices and freezing out roles that have not yet been tried (LLM as simulator, LLM as pricing engine, LLM as fairness auditor). The pre-benchmark stage may be the right stage; consolidation may be premature.

*Verdict (v2).* Real gap, but feasibility-constrained: a benchmark designed too early may calcify the role taxonomy. The best near-term project is probably a *role-comparison study* (run the same problem with the LLM in three roles and report) rather than a frozen benchmark.

### 7.3 Gap C — cross-city transfer and sim-to-real for production deployment

*Importance.* Production deployments (Xu2018, Eshkevari2022, Azagirre2024) are tied to single platforms. Cross-city transfer determines whether the academic methods (Wang2025, Hu2025, Tresca2026) translate to the next platform without per-city retraining. The importance follows from the asymmetry: there are far more cities than papers.

*Reasonableness.* Cross-city transfer is explicitly demonstrated for the bi-level AMoD line (Gammelli2021 zero-shot Chengdu↔NYC; Tresca2026 meta-RL with +18% gain on disturbance). Foundation models in the LLM line claim cross-city zero-shot for forecasting (UniST: MAE 26.84 cross-city; UrbanGPT: ~28% over generalization baselines). The gap is whether these forecasting/rebalancing transfers extend to **deployed matching policies** at production scale, where the production lineage has not reported transfer numbers.

*Feasibility.* Two paths: (a) extend Tresca2026's meta-RL across truly heterogeneous cities (not just NYC↔Brooklyn↔Shenzhen); (b) use a foundation-model encoder (UniST/UrbanGPT) as the state representation for a matching policy and test whether the foundation-model's cross-city generalization transfers to the policy. (a) is feasible with simulator data; (b) is feasible with public taxi data; both are within the scope of a 1–2 year project.

*Counter-reason* (added in v2). Cross-city transfer at the *forecasting* layer is likely to be much easier than at the *matching-policy* layer, because matching policies are tightly coupled to platform-specific operational constraints (driver pool composition, regulatory constraints, surge rules, dispatch latency budgets) that are not captured by ST data alone. A foundation-model encoder may transfer cleanly while the policy on top still requires full per-city retraining; in that case the "cross-city transfer for matching" framing of the gap collapses to "use a transferable encoder," which is already what UniST/UrbanGPT propose.

*Verdict (v2).* Real gap on the *full-pipeline transfer* axis; the *encoder-only transfer* version of the gap is already partly closed by UniST and UrbanGPT. A research project should be explicit about which version it is targeting.

A fourth candidate gap — "fairness as a hard constraint" — partially fails the *reasonableness* test because the field is already moving in this direction (JDRCL, Jusup2025, Lyu2025). It is more accurate to call fairness an emerging consensus rather than a gap. A fifth candidate — "cross-city / population not yet studied" — fails the *importance* test as written; "population X has not been studied" is the bad-gap framing flagged by the guide.

---

## 8. Operational implications for a research project entering this field

The synthesis above implies five concrete defaults for a new research project in this space.

A project should **default to the two-stage RL+combinatorial template** for matching unless it has an architectural reason to depart from it; if it does, it must show how feasibility is preserved (Theme 1). The bar to beat for end-to-end alternatives is Yue2024's verified 0.7–3.9% TDI on Didi data.

A project should **default to a graph encoder (GAT or MPNN)** for spatial state, and spend its modeling budget on the coordination geometry (driver-as-agent vs grid-as-agent) rather than the encoder choice (Themes 2 and 3). The encoder-default conclusion is scoped to the corpus reviewed and may not survive expansion to the ST-GNN forecasting literature.

A project working on dispatching at scale should **pre-decide** between driver-as-agent (cleaner driver-level metrics, harder credit assignment) and grid-as-agent (cleaner aggregate metrics, harder driver attribution) based on the metric of interest (Theme 3).

A project working on pricing should **adopt strategic agents** as a structural assumption and benchmark against the comparative-static predictions of Besbes2021 and the welfare decomposition of Castillo2025 (with corrected direction and explicit Houston/Uber scoping); it should also engage with the bounded-out MFG-RL literature as the bridge between OR equilibrium and DRL control (Section 11), not treat the two as disconnected (Theme 4, v2 corrected).

A project working on LLMs in ride-hailing should **pick one LLM role and one non-LLM baseline of the same operational class**, and report inference latency and grounding behavior; "LLM is the controller" with a 1–5 s decision latency is not a deployment-ready claim (Theme 6).

---

## 9. Reference table

The 27 papers are reproduced here in the order in which they are first cited in this review, with venue/year and a one-line characterization. The "seminal" label has been restricted in v2 to genuinely paradigm-defining works (Gammelli2021 for the bi-level GNN-RL line; Besbes2021 for spatial pricing equilibrium); other papers are described as "early," "first-of-kind," or "foundational" depending on what is defensible.

1. **Xu2018** — KDD 2018 — foundational "learning + planning" Didi dispatch baseline; 0.5–5% revenue.
2. **Eshkevari2022** — KDD 2022 — first standalone deployed RL dispatcher (Didi int'l); 1.3% A/B, up to 5.3% post-deployment.
3. **Azagirre2024** — INFORMS J. Applied Analytics 2024 — first non-Didi deployed RL match (Lyft); $30M/yr.
4. **Yang2024** — KDD 2024 — cooperative-game offline RL with goal-reaching credit; ~5–6% IORR/IGMV in City B OPE [v2 corrected].
5. **Yue2024** — CIKM 2024 — end-to-end DRL replacing two-stage; 0.7–3.9% TDI, 1–2% CR.
6. **Zhang2024 (NondBREM)** — AAAI 2024 — offline RL with nondeterministic-BCQ + REM; 3.76% response rate.
7. **Lin2018** — KDD 2018 — early contextual MARL fleet management.
8. **Li2019** — WWW 2019 — early mean-field MARL dispatch.
9. **Hao2023 (GAT-MF)** — KDD 2023 — graph-attention mean field; +42.7% perf (heterogeneous benchmarks), −86.4% time, −19.2% GPU.
10. **Wang2025 (CoopRide)** — KDD 2025 — city-scale grid-as-agent MARL; up to 12.4% over reported baselines.
11. **Hu2025 (BMG-Q)** — T-ITS 2025 — graph-attention DDQN + ILP for ride-pooling; ~10% reward, >50% overestimation reduction.
12. **Enders2023** — L4DC 2023 — early hybrid MASAC + bipartite matching template (numbers unverified in v2).
13. **Hoppe2024** — L4DC 2024 — global vs local rewards in hybrid MARL (numbers unverified in v2).
14. **Jusup2025** — arXiv 2025 — constrained MFRL with accessibility constraints at 10⁴ vehicles.
15. **Gammelli2021** — IEEE CDC 2021 — **seminal** bi-level GNN-RL+LP for AMoD.
16. **Gammelli2023** — ICML 2023 — bi-level GNN-RL generalized to network-flow control.
17. **Singhal2024** — ECC 2024 — bi-level GNN-RL on space-charge-time graph for electric AMoD.
18. **Tresca2026** — TCNS 2026 (in press, venue confirmation pending) — hierarchical bi-level GNN-RL+LP at robo-taxi scale; meta-RL.
19. **Besbes2021** — Management Science 2021 — **seminal** spatial pricing equilibrium; local myopic surge suboptimal.
20. **Afeche2023** — MSOM 2023 — strategic drivers and admission control.
21. **Castillo2025** — Econometrica 2025 — surge welfare incidence on Uber Houston data; **+3.53% total welfare with surge** [v2 corrected direction].
22. **Zhang2023** — AAAI 2023 — joint future-aware pricing+matching for ride-pooling (numbers unverified in v2).
23. **Sun2024 (JDRCL)** — TKDE 2024 (KDD'22 ext.; venue confirmation pending) — constrained MARL for joint matching+repositioning under max-min fairness.
24. **Yuan2024 (UniST)** — KDD 2024 — prompt-empowered universal ST foundation model.
25. **Li2024 (UrbanGPT)** — KDD 2024 — instruction-tuned ST-LLM for zero-shot prediction.
26. **Han2025 (GARLIC)** — AAAI 2025 — GPT-augmented RL with multiview-graph state and dynamic reward learning.
27. **Lyu2025 (LLM-ODDR)** — arXiv 2025 — LLM-as-agent for joint matching+repositioning.

---

## 10. Audit checklist (Step 4–6 verification)

- [x] **Corpus**: 27 papers across at least two databases (arXiv, ACM DL, PMLR, INFORMS, Wiley/Econometrica, T-ITS), seeded from review materials. **Acknowledged shortfall vs the guide's 30–50 target; v3 expansion plan in Section 11.**
- [x] **Search strategy**: documented in Section 1.1 — databases, keyword set, time window, inclusion criteria, and explicit bounded-out adjacent literatures with defenses.
- [x] **Schema completeness**: every paper has a structured entry in Section 2; cells without a verifiable number are marked `n/a` or `(unverified)`.
- [x] **v2 extraction-error corrections**: Castillo 2025 direction and Yang 2024 cell errors fixed in Sections 0, 2.4, 2.1, 3.1, 3.4, 5.4. Three remaining unverified numbers (Enders 2023, Hoppe 2024, Zhang 2023) are demoted from theme-anchoring to illustrative status.
- [x] **Tag hygiene**: 52 tags across six axes (Section 2.6, count corrected from v1's incorrect 18 claim); near-duplicates merged (`#joint-three-way` reserved for the unaddressed three-way case).
- [x] **Synthesis matrices**: 5 matrices (Sections 3.1–3.5), each with a single comparison axis and explicit `comparability_notes`.
- [x] **Themes**: 6 themes (2 consensus, 2 controversy, 2 gap), each backed by an explicit list of supporting and contradicting papers; topic sentences scoped to "within the corpus reviewed" where the corpus is the salient bound (Themes 1, 2).
- [x] **Gap justification**: Section 7 evaluates three candidate gaps against importance/reasonableness/feasibility **and adds at least one counter-reason per gap**; verdicts now turn on which side of the argument is stronger rather than uniformly clearing all three (v2 fix).
- [x] **MEAL paragraphs**: each theme paragraph in Section 5 has identifiable M, E, A, L; grammatical subjects are concepts/findings, not author names.
- [x] **Summary-mode audit**: no paragraph in Section 5 is structured as "Author X said... Author Y said..."; author keys appear only in parenthetical citations.
- [x] **Forward link**: Section 8 connects the synthesis to specific design choices for a new research project; Section 7 connects the gap analysis to specific feasible projects.
- [x] **"Seminal" usage**: restricted to Gammelli 2021 (paradigm-defining for bi-level GNN-RL) and Besbes 2021 (foundational for spatial pricing equilibrium); other "seminal" labels in v1 downgraded to "early," "first-of-kind," or "foundational" in v2.
- [x] **Geographic/platform scope**: explicitly flagged in Section 1.1 (Didi/Lyft/NYC concentration; Castillo Houston-only) and propagated into Theme 4 and Section 8 as scoping caveats.

---

## 11. v3 expansion plan: adjacent literatures the next-revision corpus should add

Section 1.1 bounded out five adjacent literatures with defenses. The list below names seed citations the v3 corpus should add. Inclusion of these papers will likely soften Theme 4's controversy claim (because MFG-RL is the bridge), expand Theme 1's evidence base for ride-pooling, and strengthen the strategic-agent baseline that Section 8 recommends.

1. **Mechanism-design / IC spatial pricing**: Ma, Fang, Parkes, "Spatio-temporal pricing for ride-sharing platforms" (EC 2019 / OR follow-on); related work on auction-style ride-hailing pricing.
2. **Mean-field-game-aware RL**: Guo, Hu, Xu, Zhang on MFG-RL theoretical foundations; Yang Yaodong's MFG-RL line; Saldi et al. on equilibrium-aware learning. **This is the literature whose absence makes Theme 4's v1 framing overstated.**
3. **Ride-pooling beyond BMG-Q**: Alonso-Mora et al. "On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment" (PNAS 2017); descendants on capacity-constrained matching and shared-ride pricing.
4. **Driver labor economics**: Hall, Horton, Knoepfle, "Pricing in ride-sharing platforms: Driver supply"; Cook et al., "The gender earnings gap in the gig economy" (RES 2021). Provides empirical grounding for the strategic-driver assumption.
5. **ST-GNN demand forecasting**: Li et al. "Diffusion Convolutional Recurrent Neural Network" (DCRNN, ICLR 2018); Wu et al. "Graph WaveNet" (IJCAI 2019); Yu et al. "STGCN" (IJCAI 2018). Provides the forecasting backbone that feeds into Theme 2's GNN-encoder default.

Adding 6–10 papers from this list would bring the corpus to 33–37, comfortably within the guide's 30–50 target and addressing the major coverage critique of v1.

---

*End of literature review (v2).*
