# Literature review: deep RL, GNNs, and LLMs for ride-hailing matching, pricing, and rebalancing (2018–2026) — v5

> Synthesis-mode literature review of **38 unique papers** (39 with the Sun 2024 cross-listing convention) organized into eight threads (production-scale RL dispatching; graph-MARL and mean-field; AMoD hybrid learning + optimization; pricing; LLMs / foundation models / joint optimization; mean-field-game-aware RL bridge; driver labor economics; ST-GNN forecasting backbone). The review is structured around the six-step workflow from `literature_review_guide_for_ai_agents.md`. Section 1 frames the question and documents the search strategy. Sections 2 and 3 expose the schema entries and the matrices. Sections 4 through 8 are the synthesis prose, organized by theme. Section 9 is the gap analysis. Section 10 is the audit checklist. Section 11 is the v6 expansion plan (what v5 still defers). Detailed correction histories are in Appendix B; methodological references for §6.5 are in Appendix C.

---

## 1. The question this review is answering

Ride-hailing platforms make three operational decisions on a spatiotemporal graph of a city: **whom to assign to whom (matching)**, **what to charge (pricing)**, and **where idle drivers should go (rebalancing)**. These decisions are coupled — pricing changes demand, demand shapes matching feasibility, and matching outcomes determine the post-trip idle distribution that rebalancing then has to correct. The question this review is answering is: *what does the 2017–2026 ML, OR, and labor-economics literature on these three decisions look like, separately and jointly, and where are the bridges and gaps between them?*

### 1.1 Search strategy, inclusion criteria, and bounded-out adjacent literatures

**Databases and time window.** The corpus was assembled by (a) seeding from `ride-share-deep-research.md`, an in-house deep-research compilation; (b) cross-checking against arXiv (2017–April 2026), ACM Digital Library (KDD, CIKM, WWW, AAAI, EC, SIGKDD), PMLR (ICML, AISTATS, L4DC), IEEE Xplore (CDC, ECC, T-ITS, TCNS), INFORMS journals (Management Science, MSOM, Journal on Applied Analytics), Econometrica (Wiley), Review of Economic Studies (Oxford), NBER, and PNAS. The time window is 2017–2026 (extended one year earlier than v3/v4 because Alonso-Mora et al. PNAS 2017 is now included as the OR-side ride-pooling foundation that v4's BMG-Q discussion was unmoored from).

**Keyword set.** Same as v4: ride-hailing OR/AND ride-sourcing OR/AND AMoD OR/AND robo-taxi, combined with deep reinforcement learning OR/AND MARL OR/AND graph neural network OR/AND foundation model OR/AND LLM, combined with matching OR/AND dispatching OR/AND rebalancing OR/AND repositioning OR/AND fleet management OR/AND pricing OR/AND surge. Extended in v5 with: mean-field game OR/AND MFG-RL; spatio-temporal pricing OR/AND mechanism design; driver supply OR/AND gig-economy labor; spatio-temporal graph forecasting OR/AND DCRNN OR/AND STGCN OR/AND Graph WaveNet.

**Inclusion criteria.** Peer-reviewed papers in the venues above plus arXiv preprints with claim-level evidence (a clearly stated method, a defined dataset or theoretical setting, and a numeric or analytical result). Position papers and tutorials were excluded. Where multiple versions of a paper exist (preprint + journal), the most recent peer-reviewed version that could be verified is cited. **v5 verification standard.** Entries that have been re-verified against the source PDF carry `verified_against_pdf` in Appendix A; entries newly added in v5 from the search-and-verification report carry `verified_against_abstract` until the PDFs are added to the working directory and re-checked. The two standards are tracked separately so the verification provenance is unambiguous.

**Corpus size and counting convention.** **38 unique papers**, **counted as 39 in threadwise listings under the Sun 2024 cross-listing convention** (Sun 2024 / JDRCL appears in two threads — primary placement Thread B for its constrained-MARL methodology, cross-referenced from Thread E for joint-optimization comparability). The corpus is now **comfortably within the guide's 30–50 target**. The v3/v4 acknowledged shortfall (26 unique / 27 with cross-listing) is closed in v5 by integrating 12 new papers across three previously bounded-out areas (MFG-RL bridge; driver labor; ST-GNN forecasting backbone) and three within-thread adds (Tang KDD 2019 bridging Xu 2018 to Eshkevari 2022; Ma-Fang-Parkes EC 2019 mechanism-design pricing; Alonso-Mora PNAS 2017 ride-pooling OR foundation).

**Bounded-out adjacent literatures (and why), with v5 status update.** v3/v4 bounded out five adjacent literatures with defenses; v5 brings four of them into the corpus and updates the bounded-out list accordingly.

1. **Mechanism-design and incentive-compatible spatial pricing** — *now in corpus.* Ma, Fang, Parkes EC 2019 ("Spatio-Temporal Pricing for Ridesharing Platforms") added to Thread D. The IC-pricing thread is now represented; further papers on auction-style ride-hailing pricing remain a v6 add.
2. **Mean-field-game-aware reinforcement learning** — *now in corpus, as new Thread F.* Four MFG-RL papers (Guo et al. NeurIPS 2019; Cui & Koeppl AISTATS 2021; Yang et al. ICML 2018; Saldi-Başar-Raginsky) added. Theme 4 is rewritten in v5 from "coverage gap in this corpus" to a substantive characterization of how the MFG-RL bridge literature does and does not engage with the OR/economics pricing equilibrium results.
3. **Ride-pooling / shared rides beyond BMG-Q** — *partly addressed.* Alonso-Mora et al. PNAS 2017 added as the seminal OR-side ride-pooling foundation; the Hu 2025 BMG-Q comparison is no longer un-moored on the OR side. Descendants of Alonso-Mora (Tachet et al. on shareability networks; Santi et al. on shared-mobility benefits) remain a v6 add.
4. **Driver labor economics** — *now in corpus, as new Thread G.* Hall, Horton, Knoepfle (NBER w30883, "Ride-Sharing Markets Re-Equilibrate") and Cook et al. (RES 2021, "Gender Earnings Gap in the Gig Economy") added. §8's strategic-driver recommendation is upgraded in v5 from "conditional on v5 driver-labor adds" to **unconditional** because the empirical motivation is now in the corpus.
5. **ST-GNN demand-forecasting backbone** — *now in corpus, as new Thread H.* DCRNN (Li et al. ICLR 2018), Graph WaveNet (Wu et al. IJCAI 2019), and STGCN (Yu et al. IJCAI 2018) added. Theme 2's "GNN encoder is the corpus standard" claim now rests on a base that includes the forecasting line, which strengthens the Theme but also surfaces the encoder-vs-attention debate (active in the forecasting line) that v3/v4 said was bounded out.

**Geographic and platform scope.** The empirical portion of the corpus is heavily concentrated on Didi, Lyft, NYC-taxi-derived simulators, and (with Castillo 2025 published) Uber Houston. Bolt / FreeNow (EU), Ola (India), Grab / Gojek (Southeast Asia) are not represented. Where empirical claims are presented in Sections 4–8, this geographic concentration is flagged.

---

## 2. Schema entries (Step 2 product)

The schema below holds, for each paper, the six fields specified in the guide (`population`, `method`, `key_finding`, `effect_size`, `limitations`, `stance`) plus a controlled-vocabulary tag set. Entries are grouped by thread for traceability but are referenced from Section 3 onward by `citation_key` only. Effect sizes that could not be located in the source are marked `n/a`. **Stance column convention** (carried from v4): the `stance` field describes argumentative position-taking within the corpus's space of methodological choices, not reception-evaluation.

### 2.1 Thread A — production-scale and offline RL for dispatching (7 papers; +1 in v5)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Xu2018 | KDD 2018 | Didi platform; >25M daily orders | system deployment; offline value-function learning + online Hungarian matching | Long-horizon driver-state value learning + Hungarian matching → 0.5–5% revenue improvement at production scale. | 0.5–5% revenue (deployed); ~10% completion-rate gain attributed to a **non-RL** driver-select→platform-assign architectural change | tabular value table, hand-engineered states, no fairness, single-platform, headline percentages confound RL with concurrent platform changes | supports the "learning + planning" template as the foundational deployed-RL baseline | #value-function, #bipartite-matching, #didi, #production-deployment, #real-data |
| **Tang2019 (CVNet)** [v5 added] | KDD 2019 (arXiv 2106.04493) | Didi platform; large-scale online A/B tests | semi-MDP formulation; Cerebellar Value Networks (CVNet) with distributed state-representation layer; transfer learning across cities | Semi-MDP CVNet for multi-driver order dispatching → significant gains in driver income and user-experience metrics; transfer learning further improves cross-city adaptability. | qualitative gains reported in production A/B (specific magnitudes pending PDF re-verification) | proprietary deployment evidence; transfer-learning gains depend on city pair similarity; CVNet stability requires careful state-representation design | bridges Xu 2018's "learning + planning" to Eshkevari 2022's online RL — the Thread A intermediate v3/v4 jumped over | #value-function, #semi-MDP, #didi, #production-deployment, #transfer-learning |
| Eshkevari2022 | KDD 2022 | Didi international markets | system deployment; on-policy DRL with online state-value iteration, MAB-pruned graph | Online state-value iteration → ≥1.3% A/B driver-income gain, up to 5.3% post-deployment. | ≥1.3% A/B; up to 5.3% post-deployment | exploration-exploitation pressure, single-platform, MAB pruning heuristics, dataset-shift not analytically bounded | supports online RL with safeguards as a deployable alternative to offline-only training | #online-RL, #didi, #production-deployment, #real-data, #safety-exploration |
| Yang2024 | KDD 2024 | Didi-collected real-world data, multi-city pilots | offline RL on logged data, framed as cooperative Markov game (**ODCMG**); Goal-Reaching Collaboration credit assignment; KM matcher | Modeling future-revenue impact of one region's value on others → ~5–6% IORR/IGMV in the City B OPE comparison. | ~5–6% IORR/IGMV (City B OPE table) | OPE variance ±7–10pp, goal-state tuning fragile, single-city training per run | contests prior schemes that evaluate region values independently; supports cooperative-game framing | #offline-RL, #MARL, #didi, #real-data, #value-function |
| Yue2024 | CIKM 2024 | Didi MICOD setting; ~30 km² geo-fence | end-to-end DRL (D2SN encoder–decoder); two-layer MDP | End-to-end o–d assignment → 0.7–3.9% TDI, 1–2% CR over best two-stage DRL baseline. | 0.7–3.9% TDI; 1–2% CR | ms-level latency budget, micro-view scope, evaluation only on Didi simulator/data | contests two-stage value+Hungarian as the necessary architecture | #end-to-end, #didi, #production-deployment, #real-data |
| Zhang2024 (NondBREM) | AAAI 2024 | 50,000+ vehicles, 20M orders | offline DRL with Nondeterministic-BCQ + Random Ensemble Mixture | Offline RL with nondeterministic action-space modeling → 3.76% order-response-rate improvement on logged data. | 3.76% ORR | extrapolation error remains the central bottleneck, no deployment numbers | supports offline RL as the deployment-safe alternative to online RL | #offline-RL, #real-data, #ensemble |
| Azagirre2024 (Lyft) | INFORMS J. Applied Analytics 2024 | Lyft global production; switchback experimentation | system deployment; online RL for matching with real-time driver-earnings estimation | Online matching RL with future-earnings estimation → ≥$30M/yr incremental revenue. | ≥$30M/yr (deployed) | switchback-under-spillovers identification concerns (see §6.5 with v5 refs); single-platform | supports online RL as deployable outside the Didi ecosystem; the headline number is a bounded estimate | #online-RL, #lyft, #production-deployment, #matching, #real-data |

### 2.2 Thread B — graph-based and mean-field MARL, plus ride-pooling OR foundation (9 papers; +1 in v5)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| **Alonso-Mora2017** [v5 added] | PNAS 114(3):462–467, 2017 | NYC taxi data; vehicle capacities up to 10 | dynamic trip–vehicle assignment via shareable-trip graph + ILP; "anytime optimal" algorithm | High-capacity dynamic ride-sharing → 2,000 vehicles cap-10 (or 3,000 cap-4) serve 98% of NYC demand, 2.8 min mean wait, 3.5 min mean trip delay. | 98% demand served; 2.8 min wait; 3.5 min trip delay (NYC, 13K-vehicle baseline) | OR-only (no learning); centralized assumption; static demand; no equilibrium agents | seminal OR-side ride-pooling foundation; provides the substrate the BMG-Q (Hu 2025) comparison was previously un-moored from | #ride-pooling, #ILP, #real-data, #OR-foundation |
| Lin2018 | KDD 2018 | NYC-taxi-derived simulator | contextual MARL (cDQN, cA2C) for fleet management | Contextual MARL with shared global state → outperforms rule-based and DQN baselines on order-response-rate. | n/a (relative gain only) | simulator-only, fixed agent vocabulary | supports contextual MARL as the early baseline for fleet repositioning | #MARL, #simulation-only, #rebalancing |
| Li2019 | WWW 2019 | Synthetic & semi-synthetic dispatch simulator | mean-field MARL (MF-Q / MF-AC) | Mean-field approximation → tractable thousand-agent dispatch with empirical convergence. | n/a (qualitative) | mean-field assumes interaction symmetry, simulator-only | supports mean-field MARL as the early formulation | #MARL, #mean-field, #simulation-only |
| Hao2023 (GAT-MF) | KDD 2023 | 100-agent grid + two real-data city scenarios | graph-attention weighted mean field MARL | Graph-attention-weighted mean field → 42.7% performance gain across heterogeneous benchmarks. | +42.7% (cross-task aggregate, not ride-hailing specific); −86.4% time; −19.2% GPU. **Methodological note:** ride-hailing-attributable share not separately reported. | aggregate across non-comparable tasks; interaction-strength weighting needs dense computation | supports MF + GAT hybrid as a strong default | #MARL, #mean-field, #GAT, #real-data |
| Enders2023 | L4DC 2023 | TUM AMoD simulator | hybrid MARL: per-agent SAC actor + central weighted bipartite matching | Factorized actor + central matcher → 5% avg / 17% peak profit over 20 test dates. | 5% avg / 17% peak (statistical objects from §5) | simulator-only, profit-only objective, deterministic supply | supports hybrid MASAC + bipartite matching as the early template | #MARL, #bipartite-matching, #AMoD, #simulation-only |
| Hoppe2024 | L4DC 2024 | AMoD simulator extending Enders | global-reward design + credit assignment for hybrid MADRL | COMA^scl beats LRA by 1.9% and beats greedy by 3.5% avg / up to 6% peak; gains not isolated to globalization. | 1.9% / 3.5% avg / up to 6% peak | simulator-only, attribution to globalization not isolated by ablation | contests local-reward defaults | #MARL, #global-reward, #AMoD, #simulation-only |
| Wang2025 (CoopRide) | KDD 2025 | Three real-world datasets, millions of orders; grid-as-agent | GNN-based MARL with within/cross-grid action unification | City-scale all-grid cooperation → up to 12.4% over reported baselines. | up to 12.4% (upper-tail across baseline–dataset combinations; not a fixed-baseline median) | training cost grows with full-graph coupling; "SOTA" varies by dataset | supports city-scale GNN-MARL with the magnitude qualified | #MARL, #GNN, #real-data |
| Hu2025 (BMG-Q) | T-ITS 2025 | NYC taxi trip data | localized bipartite-match graph attention DDQN + ILP coordinator | Local bipartite match graph attention + ILP → ~10% reward gain, >50% overestimation reduction (within-paper comparative). | ~+10% reward; >50% overest. ↓ (rollout-horizon-dependent measurement) | NYC-only, ride-pooling-specific, ILP cost grows with batch size | supports graph-attention DDQN + ILP for ride-pooling; the Alonso-Mora 2017 OR baseline is now in the corpus (Thread B) so the comparison is no longer un-moored | #MARL, #GAT, #bipartite-matching, #ride-pooling, #real-data |
| Jusup2025 | arXiv 2025 (under review) | Continuous-state simulator; 10⁴ vehicles | constrained mean-field RL with accessibility/fairness constraints | Constrained MFRL → scales to tens of thousands of vehicles with provable accessibility constraint satisfaction. | n/a (theoretical guarantee + simulator gains) | simulator-only, accessibility metric must be specified ex ante | extends MFRL with hard constraints | #MARL, #mean-field, #constraints, #fairness, #simulation-only |

### 2.3 Thread C — AMoD hybrid learning + optimization (4 papers; Tresca reframed as preprint in v5)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Gammelli2021 | IEEE CDC 2021 | Chengdu and NYC, 4×4 grid | bi-level GNN-RL: A2C + GCN over network-flow LP | Bi-level GNN-RL for AMoD rebalancing → 1.6–2.2% from oracle and zero-shot transfer across cities. | 1.6% NYC, 2.2% Chengdu deviation from oracle | small grid, Poisson demand, simulator-only | supports the bi-level GNN-RL+LP architecture as paradigm-defining | #GNN-RL, #bi-level, #LP, #AMoD, #rebalancing, #cross-city-transfer |
| Gammelli2023 | ICML 2023 | Synthetic min-cost flow; supply-chain; DVR | bi-level GNN-RL framework generalized | Bi-level GNN-RL for general network-flow control → 86.7%–99.5% of oracle. | 86.7%–99.5% of oracle | linear-only inner problem, large abstract networks not validated | generalizes the AMoD bi-level paradigm | #GNN-RL, #bi-level, #LP, #network-flow |
| Singhal2024 | ECC 2024 | SF and NYC; 5–20 spatial regions | bi-level GNN-RL on space-charge-time graph for electric AMoD; SAC | Space-charge-time GNN-RL → 89% of oracle (5-region NY); 100× LP speedup; 55–89% across scales. | 89% at 5-region NY; 55–89% across scales; 100× speedup | "perfect-foresight demand" baseline; charging-network configuration fixed | extends GNN-RL to electric AMoD | #GNN-RL, #bi-level, #electric-AMoD, #charging |
| Tresca2025 [v5 reframed] | preprint, arXiv 2504.06125 (April 2025), **submitted to TCNS, under review (not in press)** | NYC macroscopic 16-station; Luxembourg microscopic + mesoscopic; Brooklyn and Shenzhen real trip data | hierarchical GNN-RL+LP with online/offline/meta-RL variants; multi-fidelity evaluation | NY 16-station ≈ −4.2% from oracle; Luxembourg meso afternoon-peak ≈ −2.4%; meta-RL vs single-city baseline ≈ +61% on Brooklyn special-event scenario. | NY ≈ −4.2%; Luxembourg meso ≈ −2.4%; meta-RL ≈ +61% on Brooklyn special event; macro→meso zero-shot gap ~60% | macroscopic↔mesoscopic gap large in zero-shot; offline RL dataset dependence | demonstrates the bi-level paradigm at robo-taxi city scale across fidelities | #GNN-RL, #bi-level, #robo-taxi, #meta-RL, #cross-city-transfer |

### 2.4 Thread D — pricing (5 papers; +1 in v5; Castillo numbers corrected to published Econometrica)

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Besbes2021 | Management Science 67(3) 2021 | Theoretical fluid model with strategic spatial supply response | measure-theoretic spatial pricing; equilibrium analysis | Local myopic surge leaves money on the table → globally network-anticipating pricing strictly dominates when supply relocates. | n/a (comparative-static, sign of effect) | continuum-supply abstraction, no congestion | supports the foundational result that local-only surge is suboptimal | #spatial-pricing, #equilibrium, #fluid-model |
| Afeche2023 | MSOM 25(5) 2023 | Theoretical fluid network with strategic drivers and rejection capability | game-theoretic equilibrium analysis with admission control | Strategic-driver model with admission control → in some regimes platform should reject demand to induce repositioning. | n/a (comparative-static; regime-dependent) | strategic-rider behavior simplified, fluid-only | shows demand rejection can be optimal under strategic drivers | #strategic-drivers, #equilibrium, #fluid-model |
| **Ma-Fang-Parkes2019** [v5 added] | EC 2019 (arXiv 1801.04015) | Theoretical model of ride-sharing with drivers as IC agents | Spatio-Temporal Pricing (STP) mechanism; subgame-perfect equilibrium analysis | STP mechanism → drivers truthfully accept dispatches in SPE; outcome welfare-optimal, envy-free, IR, BB, core-selecting; **dominant-strategy version impossible**. | qualitative welfare improvement and earning-equity gains in simulation; impossibility result for DSE | continuum-driver assumption; theoretical model, not deployed | establishes that IC mechanism design and equilibrium pricing answer different questions; the STP mechanism is the bridge | #spatial-pricing, #mechanism-design, #incentive-compatible, #equilibrium |
| **Castillo2025** [v5 corrected — published Econometrica numbers replace 2019 working-paper numbers] | Castillo, "Who Benefits from Surge Pricing?" *Econometrica* 93(5):1811–1854 (Sep 2025) | Uber data, Houston market | structural empirical model of spatial equilibrium with demand, supply, and matching technology | **Surge pricing increases total welfare by 2.15% of gross revenue** relative to a uniform-pricing counterfactual; rider surplus +3.57%; driver surplus −0.98%; platform profits −0.50% (all as % of gross revenue). Riders at all income levels benefit; among drivers, those who work long hours are hurt the most, especially women. | +2.15% total welfare; +3.57% rider surplus; −0.98% driver surplus; −0.50% platform profits (all as % of gross revenue) | one-city study (Houston only); identification depends on reduced-form variation; model abstracts from rider learning. **v5 correction:** v3/v4 cited the 2019 working-paper version with numbers (+3.53%/+6.97%/−1.97%) that are *roughly double* the published Econometrica magnitudes, and did not split out platform profits. The published numbers are now the authoritative cite. | settles a canonical question on surge welfare *for the Houston market*: aggregate welfare gain comes with redistribution from drivers (especially long-hours and female drivers) to riders, with a small platform-profit cost | #surge-pricing, #welfare, #structural-empirical, #real-data |
| Zhang2023 | AAAI 2023 | NYC taxi network simulator | future-aware joint pricing-matching ADP for ride-pooling | Joint future-aware pricing+matching → average +6.4% revenue / −10.6% vehicles; max +17% / −14%. | average +6.4% / −10.6%; max +17% / −14% | simulator-only, single-pool size, no equilibrium agents | supports joint pricing+matching DRL as a viable bridge between Threads D and F (MFG-RL). | #joint-pricing-matching, #ADP, #ride-pooling, #simulation-only |

### 2.5 Thread E — LLMs, foundation models, and joint optimization (4 LLM/FM papers + Sun 2024 cross-listed; Sun venue confirmed in v5)

The Sun 2024 cross-listing convention is unchanged from v4: primary placement Thread B, cross-referenced from Thread E for joint-optimization comparability.

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| Yuan2024 (UniST) | KDD 2024 | 20+ urban ST datasets, 15 cities | ST foundation model with masked pretraining + knowledge-guided prompts | Pretraining + prompt-tuning on heterogeneous ST data → +10.1% over no-pretrain baseline. | MAE 26.84 (zero-shot); +10.1% over no-pretrain | not a controller (forecasting only) | supports the ST foundation model paradigm | #ST-foundation-model, #zero-shot, #cross-city-transfer |
| Li2024 (UrbanGPT) | KDD 2024 | NYC taxi/bike/crime ST datasets; zero-shot | LLM as encoder via ST dependency encoder + instruction tuning | Instruction-tuned ST-LLM encoder → ~28% over generalization baselines. | MAE 6.16 NYC-taxi zero-shot; ~+28% | LLM inference cost, zero-shot only | supports LLM-as-encoder for cross-domain ST prediction | #LLM, #ST-foundation-model, #zero-shot, #instruction-tuning |
| Han2025 (GARLIC) | AAAI 2025 | Manhattan taxi (4h) and Hangzhou ride-hailing (30d) | GPT-augmented RL: multiview-graph state; GPT-driven dynamic reward learning | GPT-driven reward shaping → empty-loaded rate 38.23% vs 40.71% best baseline = 2.48 pp improvement. | 2.48 pp empty-load reduction; 0.30 km error | LLM compute heavy; V2X latency 50–200 ms; single-city eval per dataset | supports the LLM-as-reward-designer template | #LLM, #GPT, #joint-matching-repositioning, #MARL, #real-data |
| Lyu2025 (LLM-ODDR) | arXiv preprint 2025 | Manhattan, Chengdu, NYC taxi/ride-hailing | LLM-as-agent for joint dispatch + repositioning; fine-tuned JointDR-GPT | LLM-as-agent for joint matching+reposition → paper-claimed +5.21% (small fleet) to +48.87% (large fleet) GMV vs traditional RL; **the +48.87% upper end is implausibly large for an unreviewed preprint with a 9× endpoint ratio over a single-paper baseline; treated as preliminary in v4/v5**. | paper-claimed +5.21–48.87% GMV; upper end downweighted | inference latency 1–5 s/decision; not peer reviewed; +48.87% upper end has not been replicated | first systematic LLM-as-agent for joint matching+repositioning; magnitudes preliminary | #LLM, #joint-matching-repositioning, #fairness, #real-data |

**Cross-referenced from Thread B for Thread E joint-optimization comparability:**

| key | venue/yr | population | method | key_finding | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| **Sun2024 (JDRCL)** [v5 venue confirmed] | "Optimizing Long-Term Efficiency and Fairness in Ride-Hailing Under Budget Constraint via Joint Order Dispatching and Driver Repositioning," **TKDE 36(7), July 2024** (extension of KDD 2022 conference paper) | Haikou, Chengdu, NYC datasets | constrained multi-objective Markov game; max-min fairness; group-based action representation; primal-dual training | Constrained max-min fairness MARL → joint matching+repositioning with convergence guarantee; consistent advantages in efficiency and fairness over baselines. | qualitative GMV gap; fairness entropy materially lifted | single-agent action-dim explosion; slow convergence; fairness vs efficiency Pareto not characterized analytically | supports constrained max-min fairness for joint matching+repositioning | #joint-matching-repositioning, #fairness, #max-min-fairness, #MARL, #real-data |

### 2.6 Thread F (NEW in v5) — mean-field-game-aware reinforcement learning (4 papers)

This thread is the bridge literature whose absence v3/v4 admitted made Theme 4's framing overstated. With these four papers in the corpus, Theme 4 is rewritten in v5 from "coverage gap in this corpus" to a substantive characterization of how MFG-RL does and does not engage with the OR/economics pricing equilibrium results.

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| **Yang2018 (Mean Field MARL)** | ICML 2018 (arXiv 1802.05438) | Resource allocation; Ising-model estimation; battle game | mean-field MARL with mean-field Q-learning and mean-field actor-critic; convergence to Nash equilibrium under regularity | Mean-field approximation reduces many-agent interaction to single-agent vs. average effect → MF-Q / MF-AC scale to populations infeasible for joint-action MARL. | qualitative — convergence + many-agent feasibility on three benchmarks | mean-field assumes interaction symmetry; convergence depends on Lipschitz/regularity assumptions; no fairness | foundational MFG-MARL methodological substrate; underlies Li 2019, Hao 2023, Jusup 2025 | #MARL, #mean-field, #MFG-RL, #foundational |
| **Guo2019 (GMFG)** | NeurIPS 2019 (arXiv 1901.09585) | Repeated ad-auction theoretical setting; stochastic games with large populations | General Mean-Field Game (GMFG) framework; existence/uniqueness of Nash equilibrium; GMF-Q learning algorithm with Boltzmann policy | GMFG framework → unique-Nash existence; naive Q-learning + fixed-point unstable; GMF-Q with Boltzmann policy is stable and convergent. | qualitative — convergence and stability superior to MARL baselines on ad-auction benchmark | repeated/stationary game assumption; ad-auction not ride-hailing; algorithm complexity high | establishes the existence and learnability of GMFG equilibria; the formal foundation for ride-hailing MFG-RL extensions | #MFG-RL, #equilibrium, #convergence, #foundational |
| **Cui-Koeppl2021** | AISTATS 2021 (arXiv 2102.01585) | Discrete-time finite MFGs; finite-horizon objectives | entropy-regularized DRL on Boltzmann fixed-point iteration for non-contractive operators | Entropy regularization + Boltzmann policies enable provable convergence to approximate fixed points where standard methods fail (non-contractive operators). | qualitative — convergence in non-contractive MFG settings | finite-horizon discrete-time setting; approximate (not exact) fixed points | shows that the contractivity assumption underlying earlier MFG-RL work (Guo 2019) often fails in practice; entropy regularization is the fix | #MFG-RL, #entropy-regularization, #convergence |
| **Saldi-Başar-Raginsky** | (e.g., "Approximate Markov-Nash Equilibria for Discrete-Time Risk-Sensitive Mean-Field Games") | Discrete-time risk-sensitive MFG, infinite-horizon | existence theorem for risk-sensitive MFG equilibrium; finite-horizon approximation argument | Existence of MFG equilibrium in risk-sensitive infinite-horizon setting; finite-horizon approximation justifies practical learning algorithms. | qualitative — existence theorem + approximation bound | risk-sensitive setting; specific cost-function class; theoretical, not empirical | provides existence guarantees that subsequent DRL approaches rely on for theoretical justification | #MFG-RL, #equilibrium, #risk-sensitive, #foundational |

### 2.7 Thread G (NEW in v5) — driver labor economics (2 papers)

These two papers are added to provide the empirical motivation for the strategic-driver assumption that §8 recommends. Without them, the recommendation rests only on OR papers (Besbes 2021; Afeche 2023) that themselves assume strategic drivers — the methodological circularity v3/v4 acknowledged.

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| **Hall-Horton-Knoepfle2023** | NBER Working Paper 30883 (Feb 2023) | Uber driver-side panel data following platform-initiated fare increases | reduced-form quasi-experimental analysis of fare changes; equilibrium adjustment dynamics | Fare increases raise per-trip earnings and initially per-hour earnings; drivers work more hours; business-stealing reduces utilization; **hourly earnings rate returns to pre-change level in roughly two months**. Riders partially compensated by shorter waits; fare increases reduce passenger welfare during sample. | hourly earnings rate re-equilibrates in ~2 months; specific quantitative magnitudes for utilization reduction reported in paper | driver-side data only; one platform; reduced-form identification | establishes that ride-sharing markets re-equilibrate quickly: drivers respond strategically to fare changes on the entry/hours margin. **This is the empirical motivation that Theme 4 and §8 require for the strategic-driver assumption to be non-circular.** | #driver-labor, #equilibrium, #strategic-drivers, #real-data, #structural-empirical |
| **Cook-Diamond-Hall-List-Oyer2021** | *Review of Economic Studies* 88(5):2210–2238 (Oct 2021) | >1M Uber rideshare drivers, US | reduced-form decomposition of gender earnings gap by experience, location preferences, driving speed | Gender earnings gap of ~7% among Uber drivers, **fully attributable** to (1) experience/learning-by-doing, (2) preferences over where to work, (3) preferences for driving speed; not customer discrimination. | ~7% gender earnings gap; full decomposition into 3 channels | one platform; US sample; identification rests on within-driver variation | shows that even algorithmically-mediated gig markets exhibit gender disparities through endogenous driver-side choices; directly relevant to §8's fairness recommendation. **Combined with Castillo 2025's finding that female drivers are most-hurt by surge, supports a fairness-aware strategic-driver framing for Gap A0.** | #driver-labor, #fairness, #gender-disparity, #real-data, #structural-empirical |

### 2.8 Thread H (NEW in v5) — ST-GNN forecasting backbone (3 papers)

These three are added because Theme 2's "GNN encoder is the corpus standard" claim previously rested on the *control-side* GNN papers only (Gammelli 2021/2023, Hao 2023, Hu 2025, Wang 2025) and the v4 hedge "may not survive expansion to the ST-GNN forecasting literature where attention vs. convolution debates are active" was honest but incomplete. With the forecasting line in the corpus, the encoder-vs-attention debate can be characterized rather than merely flagged.

| key | venue/yr | population | method | key_finding (≤25 words) | effect_size | limitations | stance | tags |
|---|---|---|---|---|---|---|---|---|
| **Yu-Yin-Zhu2018 (STGCN)** | IJCAI 2018 (arXiv 1709.04875) | METR-LA, PeMSD7 traffic datasets | spatio-temporal graph convolutional network; complete convolutional structure (no recurrence); fast training | STGCN with complete convolutional ST structure → consistently outperforms FC-LSTM and DCRNN-style baselines on multi-scale traffic forecasting with fewer parameters and faster training. | RMSE/MAE gains over FC-LSTM and DCRNN baselines (per IJCAI Table) | traffic forecasting only; convolutional inductive bias; fixed graph | establishes convolutional alternative to recurrent ST modeling; influences encoder choice in control-side GNN-RL | #ST-GNN, #GCN, #traffic-forecasting, #foundational |
| **Li-Yu-Shahabi-Liu2018 (DCRNN)** | ICLR 2018 (arXiv 1707.01926) | METR-LA, PEMS-BAY traffic datasets | diffusion convolutional recurrent neural network; bidirectional random walk + encoder-decoder + scheduled sampling | DCRNN models traffic as a diffusion process on a directed graph → SOTA on multi-step traffic forecasting; advantage grows with forecasting horizon. | best RMSE/MAE/MAPE across horizons in 2018 evaluation | traffic forecasting only; recurrent — slower than convolutional alternatives | establishes diffusion-on-graph framing; the DCRNN/STGCN debate (recurrent vs convolutional) is the encoder-vs-encoder debate Theme 2 hedged about | #ST-GNN, #diffusion-convolution, #recurrent, #traffic-forecasting, #foundational |
| **Wu-Pan-Long-Jiang-Zhang2019 (Graph WaveNet)** | IJCAI 2019 (arXiv 1906.00121) | METR-LA, PEMS-BAY | adaptive dependency matrix learned via node embedding + dilated 1D convolution | Adaptive learned graph + dilated convolution → captures hidden spatial dependency without prior graph; SOTA on PEMS-BAY/METR-LA in 2019. | best RMSE/MAE/MAPE on both benchmarks in 2019 evaluation | dilated-convolution receptive field grows with depth; adaptive matrix can overfit; traffic forecasting only | establishes that the spatial graph itself can be learned rather than imposed — directly relevant to control-side GNN-RL where the graph structure is often given by the city geometry but may not match the operationally relevant interaction structure | #ST-GNN, #adaptive-graph, #dilated-convolution, #traffic-forecasting |

### 2.9 Tag dictionary (updated for v5)

The corpus uses the following controlled vocabulary, partitioned by axis. v4 had 51 tags; v5 adds **8 new tags** for the new threads. Total: **59 distinct tags across seven strictly-partitioned axes**. New tags marked [v5].

- **Method axis (16):** `#online-RL`, `#offline-RL`, `#end-to-end`, `#value-function`, `#bipartite-matching`, `#MARL`, `#mean-field`, `#GNN-RL`, `#GAT`, `#GNN`, `#bi-level`, `#LP`, `#network-flow`, `#ADP`, `#ensemble`, `#semi-MDP` [v5].
- **LLM/FM axis (5):** `#LLM`, `#GPT`, `#ST-foundation-model`, `#instruction-tuning`, `#prompt-tuning`.
- **MFG-RL axis (NEW in v5; 4):** `#MFG-RL` [v5], `#entropy-regularization` [v5], `#risk-sensitive` [v5], `#convergence` [v5].
- **ST-GNN forecasting axis (NEW in v5; 4):** `#ST-GNN` [v5], `#diffusion-convolution` [v5], `#adaptive-graph` [v5], `#traffic-forecasting` [v5].
- **Application/operational axis (10):** `#matching`, `#rebalancing`, `#electric-AMoD`, `#charging`, `#robo-taxi`, `#AMoD`, `#ride-pooling`, `#joint-pricing-matching`, `#joint-matching-repositioning`, `#joint-three-way`.
- **Pricing/economics axis (10):** `#strategic-drivers`, `#equilibrium`, `#fluid-model`, `#structural-empirical`, `#welfare`, `#spatial-pricing`, `#surge-pricing`, `#meta-RL`, `#mechanism-design` [v5], `#incentive-compatible` [v5].
- **Constraint/objective axis (4):** `#fairness`, `#max-min-fairness`, `#constraints`, `#gender-disparity` [v5].
- **Data/deployment axis (10):** `#real-data`, `#simulation-only`, `#production-deployment`, `#didi`, `#lyft`, `#safety-exploration`, `#cross-city-transfer`, `#zero-shot`, `#global-reward`, `#transfer-learning` [v5].
- **Foundational/methodological axis (NEW in v5; 1):** `#foundational` [v5] (applies to seminal substrate papers — Yang 2018, Saldi, Alonso-Mora 2017, STGCN, DCRNN — within their respective threads).
- **Driver-labor axis (NEW in v5; 1):** `#driver-labor` [v5].

`#joint-three-way` is reserved for matching+pricing+rebalancing simultaneously and appears on no paper in the corpus, since no paper fully satisfies it.

---

## 3. Synthesis matrices (Step 3 product)

The matrices below take a *lateral* view of the corpus along five comparison axes (unchanged from v4). The new v5 papers extend rows in Matrices 1, 2, 3, and 4; a new Matrix 6 is added for the MFG-RL bridge thread.

### 3.1 Matrix 1 — matching/dispatching: data regime × action representation

*Comparison axis*: **how is the matching policy learned and what does the policy output?**

| paper | data regime | output representation | combinatorial step | deployed? | headline number | comparability_notes |
|---|---|---|---|---|---|---|
| Xu2018 | offline value table, online use | value(state) → edge weight | Hungarian | yes (Didi) | 0.5–5% revenue (RL-attributable; ~10% completion-rate from non-RL change) | reference baseline |
| **Tang2019 (CVNet)** [v5] | semi-MDP value learning + transfer | value(state) → edge weight | implicit | yes (Didi A/B) | qualitative gains (specific magnitudes pending PDF re-verification) | **Thread A bridge between Xu 2018 and Eshkevari 2022** |
| Eshkevari2022 | online state-value iteration | value(state) → edge weight | bipartite matching with smoothing & MAB pruning | yes (Didi int'l) | 1.3% / up to 5.3% income | online learning + safeguards |
| Yang2024 | offline RL on logged data | value(state) of region as future-impact term | KM (default); GS (alternative) | pilot only | ~5–6% IORR/IGMV (City B OPE) | replaces region-value independence assumption |
| Yue2024 | offline RL on Didi simulator + data | end-to-end o–d assignment | none (subsumed) | unspecified | 0.7–3.9% TDI / 1–2% CR | single-stage replacement |
| Zhang2024 (NondBREM) | offline RL only | discrete action → assignment | implicit | no | 3.76% response rate | safety-first formulation |
| Azagirre2024 (Lyft) | online RL | future-earnings → matching scores | bipartite matching | yes (Lyft global) | $30M/yr | switchback caveat (§6.5 with v5 refs) |
| Wang2025 (CoopRide) | offline + simulator | grid cooperation actions | implicit per-grid | unspecified | up to 12.4% (upper-tail across baseline–dataset combinations) | grid-as-agent |
| Hu2025 (BMG-Q) | simulator + NYC trips | per-agent Q over local match graph | ILP with posterior score | no | ~+10% reward, >50% overest. ↓ (within-paper comparative) | ride-pooling specific; **Alonso-Mora 2017 OR baseline now in corpus** |
| Enders2023 | simulator | per-agent SAC embedding → weights | weighted bipartite matching | no | 5% avg / 17% peak | hybrid template |
| Hoppe2024 | simulator | reward-shaped per-agent embedding | bipartite matching | no | 1.9% / 3.5% avg / up to 6% peak | gains not isolated to globalization |
| Lin2018 | simulator | contextual A2C / DQN per cell | implicit | no | n/a | early MARL fleet baseline |
| Li2019 | simulator | mean-field Q over neighbors | implicit | no | n/a | early MFMARL baseline |
| Hao2023 (GAT-MF) | simulator + city data | weighted-MF Q with attention | implicit | no | +42.7% (cross-task aggregate; ride-hailing-attributable share not separately reported) | scales to >3000 agents |
| Jusup2025 | continuous-state simulator | constrained MF policy | implicit | no | n/a | constraint-feasible at 10⁴ |
| Han2025 (GARLIC) | simulator + 2 city datasets | RL policy with GPT-shaped reward | bipartite matching | no | empty-load 38.23% vs 40.71% = 2.48 pp | LLM as reward designer |
| Lyu2025 (LLM-ODDR) | three city datasets (varied fleet) | LLM → assignment + reposition | implicit | no | paper-claimed +5.21–48.87% GMV (upper end preliminary) | LLM as planner |
| Sun2024 (JDRCL) | three city datasets | per-driver MARL under constraint | implicit | no | qualitative GMV gap; fairness lift | constrained joint matching+reposition |

*Comparability notes for Matrix 1.* Two-stage pipelines and end-to-end pipelines are rendered comparable only on the deployment-readiness axis; their headline numbers measure different objectives, so direct numeric comparison is not warranted.

### 3.2 Matrix 2 — MARL coordination geometry: agent definition × cooperation scope

*Comparison axis*: **what is an "agent" and how far does cooperation extend?** (Same structure as v4 with unchanged content; Yang 2018 from Thread F is the methodological substrate underlying Li 2019, Hao 2023, Jusup 2025 rows below.)

| paper | agent definition | cooperation scope | mechanism | scaling demonstrated |
|---|---|---|---|---|
| Lin2018 | grid cell | cell-local context | shared global state vector | NYC simulator |
| Li2019 | driver | mean-field (all neighbors symmetric) — substrate from Yang2018 | mean-field Q | thousands |
| Hao2023 (GAT-MF) | agent (driver-like) | weighted mean field with attention — substrate from Yang2018 | attention weights over neighbors | >3000 |
| Enders2023 | vehicle | central matcher, decentralized actor | MASAC + bipartite matching | hundreds–thousands |
| Hoppe2024 | vehicle | central matcher, reward-globalized | global reward + credit assignment | hundreds–thousands |
| Jusup2025 | vehicle | mean-field with constraints — substrate from Yang2018 | constrained MFRL | 10⁴ |
| Wang2025 (CoopRide) | grid | all-grid cooperation | GNN encoded + within/cross-grid action unification | city-scale |
| Yang2024 (ODCMG) | region | cooperative Markov game with goal-state credit | environment model + scoring + policy | city pilots |
| Sun2024 (JDRCL) | driver | constrained multi-objective game | max-min fairness + MARL | three-city scale |
| Hu2025 (BMG-Q) | vehicle (in pool) | local bipartite-match graph | GAT + ILP | thousands (NYC) |
| Lyu2025 (LLM-ODDR) | LLM-controlled vehicle | LLM-mediated coordination | prompt-driven joint reasoning | fleet sizes 100–500 |

*Comparability notes for Matrix 2.* Same as v4: papers either define the agent as **driver/vehicle** or **grid/region**. **No paper in the corpus uses the rider as an agent**, despite riders being strategic in Castillo 2025 and Besbes 2021. With Castillo 2025's published numbers showing that female and long-hours drivers are most-hurt by surge (and Cook et al. 2021 documenting a 7% gender earnings gap explained by location and driving-speed preferences), the rider-as-agent absence is now joined by a **driver-heterogeneity-as-agent absence**: no paper in the corpus models drivers as a heterogeneous population whose responses to surge differ by demographic.

### 3.3 Matrix 3 — bi-level GNN-RL + optimization paradigm: extension axis

*Comparison axis*: **what does each Pavone-lab and TUM bi-level paper extend?** (Unchanged from v4 substantively; Tresca 2025 reframed as preprint.)

| paper | extension axis | inner problem | empirical finding |
|---|---|---|---|
| Gammelli2021 | scope: introduce bi-level GNN-RL+LP for AMoD | LP rebalancing | within 1.6–2.2% of oracle; cross-city zero-shot |
| Gammelli2023 | scope: generalize to network-flow control | LP/linear inner | 86.7%–99.5% of oracle |
| Singhal2024 | physics: incorporate charging | LP on space-charge-time graph | 89% at 5-region NY (55–89% across scales); 100× speedup |
| Tresca2025 [v5: preprint, not in press] | scale + transfer: macro→meso; meta-RL | hierarchical LP | NY ≈ −4.2%; Lux meso ≈ −2.4%; meta-RL ≈ +61% on Brooklyn special event |
| Enders2023 | architecture: bring bi-level to MARL via MASAC + bipartite matching | bipartite matching | 5% avg / 17% peak |
| Hoppe2024 | reward: study global vs local rewards | bipartite matching | 1.9% / 3.5% avg / up to 6% peak |

### 3.4 Matrix 4 — pricing: methodological tradition × strategic-agent assumption

*Comparison axis*: **what tradition is the pricing analysis in, and what is assumed about strategic agents?** (Castillo numbers corrected; Ma-Fang-Parkes added.)

| paper | tradition | strategic riders? | strategic drivers? | what is solved | numeric headline |
|---|---|---|---|---|---|
| Besbes2021 | OR / equilibrium fluid | partial | yes (relocate) | optimality of network-anticipating pricing | comparative-static (sign) |
| Afeche2023 | OR / equilibrium fluid | partial | yes (reject demand) | regime conditions for admission control | comparative-static (regime-dependent) |
| **Ma-Fang-Parkes2019** [v5 added] | mechanism design / IC | implicit (price-takers) | yes (truthful acceptance) | STP mechanism welfare-optimal in SPE; impossibility in DSE | qualitative welfare/equity gains in simulation |
| **Castillo2025** [v5 corrected to published Econometrica] | structural empirical | yes (price-elastic demand, all income levels) | yes (entry, hours, gender-heterogeneous response) | welfare incidence of surge | **+2.15% total welfare; +3.57% rider; −0.98% driver; −0.50% platform profits** (all as % of gross revenue, Houston only) |
| Zhang2023 | DRL / ADP | no | no | joint future-aware pricing+matching | average +6.4%/−10.6%; max +17%/−14% |

*Comparability notes for Matrix 4.* The matrix exposes a fuller picture in v5 than in v4: the pricing literature in this corpus now spans four traditions — OR/equilibrium fluid (Besbes; Afeche), mechanism design / IC (Ma-Fang-Parkes), structural empirical (Castillo), and DRL/ADP (Zhang). The single-DRL-pricing-paper observation that motivated v3/v4's Theme 4 downgrade is unchanged; **what is now visible is that the OR/economics side has three distinct methodological traditions, and the DRL pricing line has not engaged with any of them**.

### 3.5 Matrix 5 — LLM/foundation-model role × ride-hailing operational role

(Unchanged from v4; Sun 2024 venue confirmed as TKDE 36(7) July 2024.)

| paper | role | controller? | benchmark | output |
|---|---|---|---|---|
| Yuan2024 (UniST) | forecaster | no | 20+ ST datasets, 15 cities | spatiotemporal predictions |
| Li2024 (UrbanGPT) | encoder for forecasting | no | NYC taxi/bike/crime | spatiotemporal predictions |
| Han2025 (GARLIC) | reward designer | no (RL still acts) | Manhattan, Hangzhou | shaped reward → dispatching policy |
| Lyu2025 (LLM-ODDR) | direct planner / controller | yes | Manhattan, Chengdu, NYC | dispatch + reposition decisions; **+48.87% upper end preliminary** |
| Sun2024 (JDRCL) [cross-ref] | n/a | n/a | Haikou, Chengdu, NYC | joint matching+reposition under fairness constraint |

### 3.6 Matrix 6 (NEW in v5) — MFG-RL bridge: theoretical scope × algorithmic readiness

*Comparison axis*: **how far does each MFG-RL paper get from theory toward operational ride-hailing applicability?**

| paper | theoretical scope | algorithm | algorithmic readiness for ride-hailing | bridges which OR/economics result? |
|---|---|---|---|---|
| Yang2018 (Mean Field MARL) | many-agent MARL with mean-field approximation | MF-Q / MF-AC | high — directly inherited by Li 2019, Hao 2023, Jusup 2025 | conceptually bridges Li 2019's mean-field formulation with the broader MARL literature |
| Guo2019 (GMFG) | stochastic games with large populations; existence of unique Nash | GMF-Q with Boltzmann policy | medium — ad-auction benchmark; ride-hailing application requires extension to non-stationary supply | conceptually bridges Besbes 2021's network-anticipating equilibrium with learnable equilibria |
| Cui-Koeppl2021 | discrete-time finite MFGs; non-contractive operators | entropy-regularized DRL on Boltzmann fixed-point iteration | medium — extends GMFG to settings where MFG-RL would otherwise diverge | bridges Saldi-Başar-Raginsky existence theorems with practical learning |
| Saldi-Başar-Raginsky | discrete-time risk-sensitive MFG; existence theorem | n/a (theoretical) | low (existence results, not algorithms) | provides the existence guarantees subsequent DRL approaches rely on |

*Comparability notes for Matrix 6.* The MFG-RL line is **currently at the "ad-auction and abstract benchmark" stage of empirical maturity**. None of the four papers reports a ride-hailing-specific empirical evaluation. The bridge to OR/economics ride-hailing pricing therefore exists at the *theoretical-foundations* level (Yang 2018 → Li 2019 / Hao 2023; Guo 2019 → existence of equilibria that DRL can target; Saldi → existence guarantees) but **not yet at the empirical-benchmark level**. This is the v5-revealed structure of Theme 4: the bridge is real and active in theory, but the operational ride-hailing payoff has not yet appeared.

---

## 4. Field overview (background paragraph)

The 2017–2026 ride-hailing ML and OR literature in this corpus has converged on **deep RL with GNN encoders and bipartite-matching combinatorial solvers** as the dominant matching/rebalancing stack, with a fast-growing **LLM/foundation-model** layer on top, a parallel **OR/economics pricing equilibrium** track, an emerging **mean-field-game-aware RL bridge** (now in corpus as Thread F), and an empirical **driver-labor backbone** (Thread G) that grounds the strategic-driver assumptions used by the pricing track. The empirical center of gravity in dispatching shifted from single-agent value learning (Xu 2018) through semi-MDP value networks with cross-city transfer (Tang 2019) to multi-agent RL with mean-field approximations grounded in Yang et al. 2018's foundational MF-MARL substrate (Li 2019, Hao 2023) and then to graph-attention MARL with grid-as-agent or vehicle-as-agent variants (Wang 2025, Hu 2025). In autonomous mobility-on-demand the bi-level GNN-RL + linear-program template (Gammelli 2021/2023, Singhal 2024, Tresca 2025) is dominant; in deployed dispatch, the production lineage at Didi (Xu 2018, Tang 2019, Eshkevari 2022, Yang 2024, Yue 2024) and Lyft (Azagirre 2024) is the strongest empirical base. On the pricing side, the OR/economics line spans equilibrium analysis (Besbes 2021, Afeche 2023), mechanism design (Ma-Fang-Parkes 2019), and structural-empirical estimation (Castillo 2025, **published Econometrica numbers: +2.15% total welfare with surge in Houston**), while DRL-based joint pricing is represented by a single paper (Zhang 2023). The MFG-RL bridge literature (Guo 2019, Cui-Koeppl 2021, Yang 2018, Saldi-Başar-Raginsky) provides the theoretical foundations for connecting the OR equilibrium track with DRL learning, but no paper yet reports a ride-hailing-specific MFG-RL empirical evaluation. The LLM thread has foundation models for urban ST prediction built atop the forecasting backbone (Thread H: DCRNN, Graph WaveNet, STGCN; LLM thread: Yuan 2024, Li 2024) and one peer-reviewed LLM-as-reward-designer (Han 2025) coexisting with an LLM-as-controller preprint (Lyu 2025, paper-claimed upper-bound treated as preliminary) and one peer-reviewed joint matching+repositioning under fairness constraints (Sun 2024, TKDE confirmed).

---

## 5. Themes (Step 4 product) and MEAL paragraphs (Step 6 product)

Six themes drawn from the matrices: two consensus, **two** controversies (Theme 4 upgraded from v4's "coverage gap" to a substantive controversy now that Thread F is in the corpus), and **two** gap themes (Theme 5 + Theme 6).

### 5.1 Theme 1 (consensus, with active contestation) — RL-proposed weights plus a combinatorial matcher is the dominant matching architecture in this corpus, but it is no longer uncontested

**Main idea.** The most common architectural template for matching is "RL learns edge weights on the driver–order bipartite graph; a combinatorial step (Hungarian, ILP, or Gale–Shapley) takes those weights and produces a feasible assignment" — but two recent papers (Yue 2024, Lyu 2025) explicitly contest the template, and the contestation is thin (one peer-reviewed paper on Didi data; one preprint with paper-claimed magnitudes whose upper end is implausible). The right framing is "dominant template, early contestation." **Evidence.** This template appears in every deployed system in the corpus — Didi's learning + planning approach (Xu 2018), Didi's CVNet semi-MDP value-network approach (Tang 2019, the Thread A bridge added in v5), Didi's online state-value iteration (Eshkevari 2022), Lyft's online matching RL (Azagirre 2024) — and in the strongest non-deployed graph-MARL baselines (Wang 2025; Hu 2025, where the combinatorial step is an ILP with a learned posterior score and the Alonso-Mora 2017 OR baseline is now in the corpus to anchor the comparison). The bi-level AMoD line (Gammelli 2021, Gammelli 2023, Singhal 2024, Tresca 2025) follows the same template with an LP rather than a bipartite-matching CO solver. Two papers contest it: Yue 2024 ("D2SN") replaces the two-stage pipeline with a single end-to-end DRL model and reports verified 0.7–3.9% TDI gains on Didi data; Lyu 2025 has the LLM directly emit assignment+reposition actions with paper-claimed gains (preliminary upper end). **Analysis.** The persistence of the two-stage template is best explained by feasibility: the matching step has hard constraints that the RL action space cannot easily encode, while the combinatorial solver guarantees feasibility regardless of how noisy the learned weights are. End-to-end approaches inherit the burden of feasibility through architectural tricks. **Link.** A current research project should default to the two-stage template unless it has a specific architectural reason to depart from it; the bar to beat for end-to-end alternatives is Yue 2024's verified 0.7–3.9% TDI on Didi data.

### 5.2 Theme 2 (consensus, broader evidence base in v5) — graph encoders are the standard state representation for spatial dependence, with the encoder-vs-attention debate now visible

**Main idea.** Within this corpus, the spatial state of a ride-hailing system is routinely encoded with graph neural networks; v5 adds the ST-GNN forecasting backbone (Thread H) which both strengthens the consensus *and* makes the encoder-vs-attention debate (DCRNN's recurrence; STGCN's complete convolution; Graph WaveNet's adaptive learned graph) explicitly visible — v4's hedge that the encoder-choice debate "may not survive expansion to the ST-GNN forecasting literature" is now obsolete because the expansion has happened. **Evidence.** Control-side: Gammelli 2021 (GCN); Gammelli 2023 (MPNN/GCN/GAT); Hao 2023 (GAT, with the methodological caveat that the 42.7% headline aggregates across heterogeneous benchmarks); Hu 2025 (graph-attention DDQN over a localized bipartite match graph); Wang 2025 (city-scale GNN, with the caveat that 12.4% is an upper-tail across baseline–dataset combinations). Forecasting-side (new in v5): DCRNN models traffic as a diffusion process on a directed graph with bidirectional random walks (Li-Yu-Shahabi-Liu 2018); STGCN replaces recurrence with complete convolution and reports faster training with fewer parameters (Yu-Yin-Zhu 2018); Graph WaveNet learns the spatial graph adaptively rather than imposing a fixed road-network graph (Wu et al. 2019). LLM/FM: UniST and UrbanGPT use the spatiotemporal graph as the substrate for foundation-model pretraining (Yuan 2024, Li 2024). **Analysis.** Two structural facts make GNNs an attractive default: cities are explicitly graph-structured, and ride-hailing decisions exhibit spatial locality at the right scale for message-passing depth of 1–3 hops. The forecasting line surfaces a substantive debate that the control-side line had largely papered over: **whether the spatial graph should be imposed (DCRNN, STGCN) or learned (Graph WaveNet)**. Current control-side practice imposes (city zones, hexagons), but forecasting-side evidence suggests learned graphs can capture interactions the imposed graph misses — a research direction the control line has not yet explored. **Link.** A new ride-hailing controller can adopt graph attention (GAT) or message-passing (MPNN) as the default encoder; if the project has reason to suspect that the operationally relevant interaction graph differs from the city geometry (e.g., ride-pooling, where shareability is the relevant graph rather than physical adjacency — Alonso-Mora 2017), it should consider an adaptive-graph approach in the spirit of Graph WaveNet.

### 5.3 Theme 3 (controversy) — agent definition: driver/vehicle vs. grid/region, with v5 surfacing a third absent option (driver heterogeneity)

**Main idea.** The corpus is split on whether the basic MARL agent should be a **driver/vehicle** or a **grid/region**, with cascading consequences for credit assignment, action space size, and how cooperation is modeled. v5 surfaces a third absent option: **drivers as a heterogeneous population**, motivated by Castillo 2025's published finding that female and long-hours drivers are most-hurt by surge, and Cook et al. 2021's documentation of a 7% gender earnings gap explained by location and driving-speed preferences. **Evidence.** Driver-as-agent: Li 2019 (mean-field, on Yang 2018's substrate), Eshkevari 2022, Enders 2023, Hoppe 2024, Hu 2025, Jusup 2025, Lyu 2025. Grid-as-agent: Wang 2025, Hao 2023's grid-world variant, Yang 2024 (ODCMG), Lin 2018. Driver-heterogeneity-as-agent: **no paper in the corpus**. The OR/empirical evidence suggests this is not a benign omission: drivers respond differently to surge (Castillo 2025) and to the equilibrium dynamics that follow fare changes (Hall-Horton-Knoepfle 2023). **Analysis.** The driver-as-agent formulation is closer to the operational reality but the joint-action space grows with fleet size. The grid-as-agent formulation collapses the action space but creates a new credit-assignment problem; Yang 2024 makes this explicit (the ODCMG formulation is exactly the question of how a region's value affects others). The driver-heterogeneity-as-agent absence reflects a corpus-wide conflation of "agent-level fairness" (max-min over drivers — Sun 2024, Jusup 2025) with "demographic fairness" (gender-aware income distribution — Cook et al. 2021); the former has been added as a constraint, while the latter has not. **Link.** A research project should pick the agent definition by what its evaluation metric looks like, and should **explicitly decide whether to model driver heterogeneity** rather than treating drivers as exchangeable; the v5-added driver-labor literature provides the empirical motivation for treating heterogeneity as a first-class concern rather than as a post-hoc fairness audit.

### 5.4 Theme 4 (controversy, upgraded in v5 from v4's "coverage gap") — DRL pricing engages with neither the OR/economics equilibrium tradition nor the MFG-RL bridge, despite both literatures providing the theoretical scaffolding

**Main idea.** With Thread F (MFG-RL bridge) and Ma-Fang-Parkes 2019 (mechanism design) now in the corpus, Theme 4 is no longer a coverage gap — it is a substantive controversy. **The OR/economics pricing literature spans three distinct traditions (equilibrium analysis: Besbes 2021, Afeche 2023; mechanism design: Ma-Fang-Parkes 2019; structural empirical: Castillo 2025), and the MFG-RL bridge literature has built theoretical foundations for connecting these to DRL (Yang 2018, Guo 2019, Cui-Koeppl 2021, Saldi-Başar-Raginsky). Yet the single DRL pricing paper in the corpus (Zhang 2023) does not engage with any of these.** **Evidence.** OR/equilibrium: local myopic surge is suboptimal (Besbes 2021); admission control can be optimal (Afeche 2023); Castillo 2025's published Econometrica result on Uber Houston is +2.15% total welfare with surge, decomposed as +3.57% rider / −0.98% driver / −0.50% platform profits (as % of gross revenue) — about half the magnitude of v3/v4's working-paper-cited numbers, with platform profits now separately reported. Mechanism design: Ma-Fang-Parkes 2019's Spatio-Temporal Pricing mechanism is welfare-optimal in subgame-perfect equilibrium with truthful driver acceptance, and the dominant-strategy version is impossible — a result that constrains the design space for any pricing controller that wants IC properties. MFG-RL bridge: Guo 2019 establishes existence and learnability of GMFG equilibria; Cui-Koeppl 2021 shows entropy regularization fixes non-contractive operators that defeat naive Q-learning; Yang 2018 provides the methodological substrate underlying the corpus's mean-field MARL line. DRL pricing: Zhang 2023 (verified average +6.4% revenue / −10.6% vehicles) is on a NYC simulator without strategic agents, no equilibrium characterization, no IC analysis, and no benchmarking against Castillo 2025's welfare decomposition. **Analysis.** The disconnect is now substantive rather than coverage-driven: three OR traditions and a mature MFG-RL theoretical line all exist, and the single DRL pricing paper has not picked up any of them. The operational gap (Matrix 6) is that **MFG-RL is at the abstract-benchmark stage of empirical maturity; no paper in Thread F reports a ride-hailing evaluation**. The bridge therefore exists in theory but has not yet produced operational ride-hailing payoffs. **Link.** A new pricing project should adopt the strategic-driver assumption (now empirically grounded by Hall-Horton-Knoepfle 2023 and Cook et al. 2021 in Thread G), benchmark against Castillo 2025's published welfare decomposition (with Houston/Uber scoping), engage explicitly with Ma-Fang-Parkes 2019's IC impossibility result if IC properties are desired, and connect to the MFG-RL line (Guo 2019 / Cui-Koeppl 2021) rather than treating ML and OR as disconnected. *The specific scoped research question is described in Gap A0 (§7.1), now updated with Castillo 2025's published numbers.*

### 5.5 Theme 5 (gap) — true three-way joint optimization of matching + pricing + rebalancing

**Main idea.** No paper in the corpus addresses all three operational levers simultaneously with peer-reviewed evidence. **Evidence.** Pairwise joint optimization exists: matching + repositioning under fairness (Sun 2024 TKDE 36(7) JDRCL); matching + repositioning by LLM agent (Lyu 2025, magnitudes preliminary); pricing + matching via ADP (Zhang 2023). The Pavone-lab AMoD line couples matching and rebalancing (Gammelli 2021, Singhal 2024, Tresca 2025) but treats pricing as exogenous. The MARL lines treat pricing as exogenous as well. **Analysis.** Two reasons explain the gap. First, matching/rebalancing levers act on second-to-minute timescales while pricing acts on minute-to-longer timescales; coupling them in a single MDP inflates state and action spaces. Second, pricing changes demand, which changes the post-matching idle distribution — a three-way joint policy must internalize that feedback, which the equilibrium literature handles via fluid models but DRL has only partially absorbed. **Link.** A research project should (a) work in pairwise matching+rebalancing first with a max-min fairness or accessibility constraint (Sun 2024, Jusup 2025); (b) borrow the strategic-driver fluid abstraction from Besbes 2021 / Afeche 2023 as the third decision module; and (c) evaluate the joint policy against Castillo 2025's published welfare incidence (rider +3.57% / driver −0.98% / platform −0.50% as % of gross revenue) to test whether three-way joint optimization changes the distributional answer; (d) engage with Ma-Fang-Parkes 2019's IC impossibility result if any version of incentive-compatibility is targeted.

### 5.6 Theme 6 (gap) — LLM role in ride-hailing is unstandardized, with the corpus too small to support standardization

**Main idea.** The LLM/foundation-model thread is the most rapidly growing line in the corpus, but the role an LLM plays in the pipeline (encoder, forecaster, reward designer, planner, simulator) is not yet standardized. With four LLM/FM papers, the lack of standardization is partly a sample-size artifact. **Evidence.** Forecaster: UrbanGPT and UniST (Yuan 2024, Li 2024) — peer-reviewed cross-city zero/few-shot generalization; the underlying ST-GNN forecasting backbone (DCRNN, STGCN, Graph WaveNet, now in Thread H) provides the substrate. Reward designer: GARLIC (Han 2025) — 2.48 pp empty-load reduction. Direct planner: LLM-ODDR (Lyu 2025) — paper-claimed +5.21–48.87% GMV; **the +48.87% upper end is preliminary**, has not been replicated, and 1–5 s/decision inference latency is a separate deployment concern. **Analysis.** Role mismatch makes head-to-head comparison impossible; a 28% encoder gain is not commensurable with a 2.48 pp reward-designer gain. No benchmark in the corpus standardizes city, fleet size, and metric the way the AMoD line did with the macroscopic 16-station simulator. The right pre-benchmark step is probably a role-comparison study (run the same problem with the LLM in three roles) rather than a frozen benchmark, exactly because the role taxonomy itself is not yet stable. **Link.** A new project should pick a single LLM role and benchmark against a non-LLM baseline of the same operational class; the Lyu 2025 +48.87% figure should not be treated as the bar to beat.

---

## 6. Where the synthesis matrix exposed surprises (corpus-specific)

Three patterns from v4 carry over; v5 adds one more.

A first surprise (carried from v4) is that **no paper in the corpus uses the rider as the agent**, despite riders being treated as strategic in Castillo 2025 (price-elastic demand, with riders at all income levels benefiting from surge) and Besbes 2021 / Ma-Fang-Parkes 2019 / Afeche 2023.

A second surprise (carried from v4) is that **the bi-level paradigm is so dominant in AMoD that the controversy is no longer "RL vs OR" but "which inner problem"** — Gammelli's lab uses an LP; the TUM lab uses bipartite matching.

A third surprise (carried from v4) is that **fairness has only just entered the corpus as a first-class objective** — JDRCL, Jusup 2025, Lyu 2025.

A fourth surprise (NEW in v5) is that **the MFG-RL bridge literature is at the abstract-benchmark stage of empirical maturity; no paper in Thread F reports a ride-hailing evaluation**. Thread F's Matrix 6 makes the gap visible: the four MFG-RL papers operate on ad-auction, Ising-model, battle-game, or abstract MFG benchmarks. The OR/economics ride-hailing track and the DRL ride-hailing track therefore have a theoretical bridge (Thread F provides it) but no empirical bridge — Theme 4's controversy is real, and Gap A0's scoped research question is exactly the empirical bridge that would close it.

### 6.5 Methodological caveats common across the corpus (with v5 references)

v3/v4 stated four caveats as the reviewer's concerns; v5 cites the methodological references that ground each caveat (these are added as Appendix C, separate from the main corpus).

**Switchback identification under spatial spillovers.** Azagirre 2024 (Lyft) reports its $30M/yr from production switchback experiments. Switchbacks alternate treatment across time blocks within a single market and assume that within-block effects are local and additive. Both assumptions are stressed in ride-hailing settings: spatial spillovers and time-varying treatment effects bias estimators. The design-of-experiments literature has documented this — see Bojinov & Simchi-Levi (2023) on the design and analysis of switchback experiments under temporal carryover; Glynn, Johari & Rasouli (2020) on adaptive experimental design with temporal interference modeled as a Markov-chain steady-state-difference problem; and Hu & Wager (2022) on switchback experiments under geometric mixing, which shows that standard switchback designs suffer carryover bias decaying as T^{-1/3} rather than T^{-1/2}, with burn-in periods recovering close-to-optimal rates. The Lyft headline number should therefore be read as a bounded estimate with non-trivial residual identification risk.

**Off-policy evaluation variance.** Yang 2024 explicitly flags OPE variance at the per-paper level (±7–10pp). Across the corpus, OPE drives several headline numbers (Yang 2024 ~5–6%; Yue 2024 0.7–3.9% TDI). Readers should weight prospective-deployment numbers (Xu 2018, Tang 2019, Eshkevari 2022, Azagirre 2024) more heavily than OPE numbers when assessing the literature's evidentiary base.

**Publication bias in deployed-RL reports.** Production deployments that fail to ship are not published; the corpus's deployed-system papers (Xu 2018, Tang 2019, Eshkevari 2022, Azagirre 2024) are without exception their respective platforms' own success reports.

**Concurrent-improvement confounding.** Xu 2018 explicitly notes that the ~10% completion-rate gain it cites came from a non-RL architectural change. This is a corpus-wide concern: production deployments ship many things at once.

These four caveats apply most directly to the deployed-RL line but also affect the bi-level AMoD line's claim of generalization (Tresca 2025's macro→meso zero-shot gap is large precisely because cross-fidelity transfer is hard) and the LLM line's reported gains. The per-paper methodological notes in v4 (Wang 2025, Hao 2023, Hu 2025) are paper-specific instances of the same general concerns; v5 retains them.

---

## 7. Gap analysis (Step 5 product)

Four candidate gaps are evaluated below. Gap A0 (load-bearing) is updated with Castillo 2025's published numbers; Gap A, Gap B, Gap C are largely carried from v4 with updated cross-references.

### 7.1 Gap A0 (load-bearing, v5 updated with published Castillo numbers) — structural alignment of a three-way DRL welfare decomposition with Castillo 2025's published equilibrium target, on a publicly-available proxy market

*Importance.* This is the single specific research question every preceding theme converges on. A successful DRL three-way controller whose welfare *structure* aligns with Castillo 2025's published four-way decomposition (positive total welfare; positive rider surplus; negative driver surplus; negative platform profits, all as % of gross revenue) — even if absolute numbers differ — would settle whether DRL can match, exceed, or distort the welfare incidence the OR equilibrium model produced. The MFG-RL bridge literature (Thread F) provides the theoretical framework, and Hall-Horton-Knoepfle 2023 + Cook et al. 2021 (Thread G) provide the empirical motivation for treating drivers as a heterogeneous strategic population.

*Reasonableness.* Castillo 2025 (now published in *Econometrica* 93(5):1811–1854) provides the welfare decomposition: +2.15% total welfare, +3.57% rider, −0.98% driver, −0.50% platform profits, with the additional finding that female drivers and long-hours drivers are most-hurt — a finding that connects directly to Cook et al. 2021's gender earnings gap decomposition. Sun 2024 (TKDE 36(7) confirmed) showed constrained multi-objective MARL is convergent in the pairwise case; Zhang 2023 showed joint pricing+matching DRL is feasible on a NYC simulator. Guo 2019 / Cui-Koeppl 2021 provide the MFG-RL theoretical scaffolding.

*Feasibility (carried from v4 with v5 update).* Houston Uber data are not public. v5 commits to **NYC taxi data with Houston-calibrated demand-elasticity and driver-heterogeneity model derived from Castillo 2025's published structural estimates**, with the calibration procedure being to fix demand elasticities, entry/hours elasticities, and a stylized gender-heterogeneity parameter (calibrated to Cook et al. 2021's earnings-gap decomposition) at the published point estimates and re-fit a NYC market-clearing simulator. This is a *structural-alignment* benchmark — it tests whether the DRL controller's four-way welfare structure (sign and rough proportion across riders, drivers, platform, and total) matches Castillo's, **not** whether the DRL controller produces the same numbers. Feasibility budget: 1–2 years.

*Counter-reason.* A NYC market under Houston-derived elasticities is neither NYC nor Houston. The benchmark therefore tests robustness of welfare structure to a transplanted demand model, not match to Castillo's actual Houston market. The result is weaker than v3/v4's "Houston benchmark" framing implied, and v5 names this honestly. The driver-heterogeneity calibration introduces an additional source of model risk.

*Verdict (v5).* Real, scoped, falsifiable gap, with feasibility re-stated as a structural-alignment exercise on a single specific proxy market. The **published-Castillo-numbers update** in v5 closes the v3/v4 working-paper-version vulnerability the prof flagged.

### 7.2 Gap A — three-way joint optimization with strategic-agent guarantees (carried from v4)

*Importance.* Filling this gap meaningfully changes the answer to the core platform-control question, because the equilibrium literature has shown pricing and rebalancing are coupled through strategic supply (Besbes 2021, Afeche 2023), and Castillo 2025 has shown the welfare incidence is heterogeneous across rider and driver subgroups.

*Reasonableness.* JDRCL's convergence guarantee for two-way joint matching+repositioning under max-min fairness shows three-way extension is a recognizable technical question. Ma-Fang-Parkes 2019's IC impossibility result constrains the design space.

*Feasibility.* Two paths: (a) couple a fluid-model pricing module with a DRL matching+repositioning module; (b) single MARL formulation with prices as agent actions on a coarser timescale. Both are feasible with current methods.

*Counter-reason* (carried from v4). The OR/economics fluid-model line has solved versions of three-way joint optimization analytically. The right framing of Gap A may be "the ML literature should adopt the equilibrium framing rather than reinvent three-way coupling in MDPs."

*Verdict (v5).* Real gap on the *implementation-as-DRL* axis; partly closed on the *analytical solvability* axis. Gap A0 is the scoped specialization v5 anchors on.

### 7.3 Gap B — LLM role taxonomy and shared benchmark (carried from v4)

*Verdict (v5).* Real gap, but feasibility-constrained. Best near-term project is a *role-comparison study*, not a frozen benchmark. (Reasoning unchanged from v4.)

### 7.4 Gap C — cross-city transfer and sim-to-real for production deployment (carried from v4)

*Verdict (v5).* Real gap on the *full-pipeline transfer* axis; *encoder-only transfer* version partly closed by UniST and UrbanGPT. (Reasoning unchanged from v4. Tang 2019's CVNet transfer-learning across cities, now in Thread A, adds a deployed-RL data point: cross-city transfer is an existing concern in the production lineage.)

A fifth candidate gap — "fairness as a hard constraint" — is now a stronger emerging-consensus pattern with v5's Thread G additions. JDRCL, Jusup 2025, Lyu 2025 give it a constraint formulation; Cook et al. 2021 grounds it empirically in gender disparity; Castillo 2025 connects it to surge welfare incidence. It is more accurate to call this an emerging consensus than a gap.

---

## 8. Operational implications for a research project entering this field

The synthesis above implies five concrete defaults for a new research project; the load-bearing research question is in §7.1.

A project should **default to the two-stage RL+combinatorial template** for matching unless it has an architectural reason to depart from it; the bar to beat for end-to-end alternatives is Yue 2024's verified 0.7–3.9% TDI on Didi data (not Lyu 2025's paper-claimed +48.87% upper end, which is preliminary).

A project should **default to a graph encoder (GAT, MPNN, or — if the operationally relevant interaction graph differs from the city geometry — adaptive-graph approaches in the spirit of Graph WaveNet)**, and spend its modeling budget on the coordination geometry rather than the encoder choice. The forecasting-side encoder debate (DCRNN's recurrence vs STGCN's complete convolution vs Graph WaveNet's adaptive learned graph) is a richer source of design intuition than v3/v4's hedge implied.

A project working on dispatching at scale should **pre-decide** between driver-as-agent, grid-as-agent, and **driver-heterogeneity-as-agent** (the v5-surfaced third option, motivated by Cook et al. 2021's gender earnings gap decomposition and Castillo 2025's published finding that female and long-hours drivers are most-hurt by surge). The corpus does not yet contain driver-heterogeneity-as-agent papers; doing one would be a corpus-novel contribution.

A project working on pricing should **adopt strategic-driver assumptions** and benchmark against the comparative-static predictions of Besbes 2021, the IC impossibility result of Ma-Fang-Parkes 2019, and the **published Castillo 2025 welfare decomposition (+2.15% total welfare; +3.57% rider; −0.98% driver; −0.50% platform profits, with female and long-hours drivers most-hurt; Houston/Uber)**. The strategic-driver assumption is now empirically grounded in the corpus by Hall-Horton-Knoepfle 2023 (drivers re-equilibrate hours within ~2 months following fare changes) and Cook et al. 2021 (7% gender earnings gap explained by location and driving-speed preferences) — **the v4 conditional framing is upgraded to unconditional in v5 because the empirical motivation is no longer bounded out of the corpus**. A project should also engage with the MFG-RL bridge literature (Guo 2019, Cui-Koeppl 2021) rather than treating ML and OR as disconnected.

A project working on LLMs in ride-hailing should **pick one LLM role and one non-LLM baseline of the same operational class**, report inference latency and grounding behavior, and not treat Lyu 2025's +48.87% paper-claimed upper end as the bar to beat. A *role-comparison study* is the right pre-benchmark contribution.

Production-deployment claims should be evaluated against the §6.5 caveats (now properly cited to Bojinov & Simchi-Levi 2023, Glynn-Johari-Rasouli 2020, and Hu & Wager 2022 for the switchback identification concerns) rather than taken at face value.

---

## 9. Reference table

The 38 unique papers (Sun 2024 cross-listed in two threads; total threadwise count 39) are listed below in the order in which they are first cited. Methodological references for §6.5 are in Appendix C. The "seminal" label is restricted to genuinely paradigm-defining works.

1. **Xu2018** — KDD 2018 — foundational "learning + planning" Didi dispatch; 0.5–5% revenue.
2. **Tang2019 (CVNet)** [v5 added] — KDD 2019 (arXiv 2106.04493) — semi-MDP value-network with CVNet; transfer learning across cities; Thread A bridge between Xu 2018 and Eshkevari 2022.
3. **Eshkevari2022** — KDD 2022 — first standalone deployed RL dispatcher (Didi int'l); 1.3% A/B, up to 5.3%.
4. **Azagirre2024** — INFORMS J. Applied Analytics 2024 — first non-Didi deployed RL match (Lyft); $30M/yr (switchback caveat).
5. **Yang2024** — KDD 2024 — cooperative-game offline RL (ODCMG); ~5–6% IORR/IGMV.
6. **Yue2024** — CIKM 2024 — end-to-end DRL; 0.7–3.9% TDI, 1–2% CR.
7. **Zhang2024 (NondBREM)** — AAAI 2024 — offline RL; 3.76% ORR.
8. **Lin2018** — KDD 2018 — early contextual MARL fleet management.
9. **Li2019** — WWW 2019 — early mean-field MARL dispatch (substrate: Yang 2018).
10. **Hao2023 (GAT-MF)** — KDD 2023 — graph-attention mean field; +42.7% (cross-task aggregate).
11. **Wang2025 (CoopRide)** — KDD 2025 — city-scale grid-as-agent MARL; up to 12.4% (upper-tail).
12. **Hu2025 (BMG-Q)** — T-ITS 2025 — graph-attention DDQN + ILP for ride-pooling (OR baseline now in corpus: Alonso-Mora 2017).
13. **Enders2023** — L4DC 2023 — early hybrid MASAC + bipartite matching; 5% avg / 17% peak.
14. **Hoppe2024** — L4DC 2024 — global vs local rewards in hybrid MARL; 1.9% / 3.5% avg / up to 6% peak.
15. **Jusup2025** — arXiv 2025 — constrained MFRL with accessibility constraints at 10⁴.
16. **Alonso-Mora2017** [v5 added] — PNAS 114(3):462–467 — **seminal** OR-side ride-pooling foundation; 98% NYC demand served at 2.8 min mean wait.
17. **Gammelli2021** — IEEE CDC 2021 — **seminal** bi-level GNN-RL+LP for AMoD.
18. **Gammelli2023** — ICML 2023 — bi-level GNN-RL generalized.
19. **Singhal2024** — ECC 2024 — bi-level GNN-RL for electric AMoD.
20. **Tresca2025** [v5 reframed: preprint, not in press] — arXiv 2504.06125, submitted to TCNS — hierarchical bi-level GNN-RL+LP at robo-taxi scale.
21. **Besbes2021** — Management Science 67(3) 2021 — **seminal** spatial pricing equilibrium.
22. **Afeche2023** — MSOM 25(5) 2023 — strategic drivers and admission control.
23. **Ma-Fang-Parkes2019** [v5 added] — EC 2019 (arXiv 1801.04015) — Spatio-Temporal Pricing mechanism; SPE welfare-optimal; DSE impossible.
24. **Castillo2025** [v5 corrected to published Econometrica] — *Econometrica* 93(5):1811–1854 (Sep 2025) — surge welfare incidence on Uber Houston; +2.15% total welfare with surge; +3.57% rider, −0.98% driver, −0.50% platform profits (all as % of gross revenue).
25. **Zhang2023** — AAAI 2023 — joint future-aware pricing+matching; average +6.4%/−10.6%; max +17%/−14%.
26. **Sun2024 (JDRCL)** [v5 venue confirmed: TKDE 36(7) Jul 2024, KDD 2022 ext.] — constrained MARL for joint matching+repositioning under max-min fairness with budget constraint.
27. **Yuan2024 (UniST)** — KDD 2024 — prompt-empowered universal ST foundation model.
28. **Li2024 (UrbanGPT)** — KDD 2024 — instruction-tuned ST-LLM.
29. **Han2025 (GARLIC)** — AAAI 2025 — GPT-augmented RL; 2.48 pp empty-load reduction.
30. **Lyu2025 (LLM-ODDR)** — arXiv 2025 — LLM-as-agent for joint matching+repositioning; +5.21–48.87% GMV (upper end preliminary).
31. **Yang2018 (Mean Field MARL)** [v5 added — Thread F] — ICML 2018 (arXiv 1802.05438) — foundational mean-field MARL; methodological substrate for Li 2019, Hao 2023, Jusup 2025.
32. **Guo2019 (GMFG)** [v5 added — Thread F] — NeurIPS 2019 (arXiv 1901.09585) — General Mean-Field Game framework with GMF-Q learning; existence of unique Nash equilibrium.
33. **Cui-Koeppl2021** [v5 added — Thread F] — AISTATS 2021 (arXiv 2102.01585) — entropy-regularized DRL for non-contractive MFG operators.
34. **Saldi-Başar-Raginsky** [v5 added — Thread F] — discrete-time risk-sensitive MFG existence and finite-horizon approximation.
35. **Hall-Horton-Knoepfle2023** [v5 added — Thread G] — NBER w30883 — Uber driver-side equilibrium adjustment to fare changes; hourly-earnings re-equilibration in ~2 months.
36. **Cook-Diamond-Hall-List-Oyer2021** [v5 added — Thread G] — *Review of Economic Studies* 88(5):2210–2238 — 7% gender earnings gap among Uber drivers, fully decomposed into experience, location preferences, driving-speed preferences.
37. **Yu-Yin-Zhu2018 (STGCN)** [v5 added — Thread H] — IJCAI 2018 (arXiv 1709.04875) — spatio-temporal graph convolutional network with complete convolutional structure.
38. **Li-Yu-Shahabi-Liu2018 (DCRNN)** [v5 added — Thread H] — ICLR 2018 (arXiv 1707.01926) — diffusion convolutional recurrent neural network.
39. **Wu-Pan-Long-Jiang-Zhang2019 (Graph WaveNet)** [v5 added — Thread H] — IJCAI 2019 (arXiv 1906.00121) — adaptive learned graph + dilated convolution.

---

## 10. Audit checklist (Step 4–6 verification)

### 10.1 Externally-verifiable items

- [x] **Corpus**: 38 unique papers (39 with Sun 2024 cross-listing) across at least two databases (arXiv, ACM DL, PMLR, INFORMS, Econometrica/Wiley, T-ITS, NBER, RES, PNAS). **Comfortably within the guide's 30–50 target — v5 closes the v3/v4 shortfall.**
- [x] **Search strategy documented**: Section 1.1 specifies databases, keyword set, time window (extended to 2017–2026 in v5), inclusion criteria, and bounded-out adjacent literatures with v5 status updates.
- [x] **Schema completeness**: every paper has a structured entry; cells without a verifiable number are marked `n/a`.
- [x] **Per-claim verification log**: every headline number has a corresponding row in Appendix A. v5-added entries carry `verified_against_abstract` until PDFs are added to the working directory; v3/v4 entries retain `verified_against_pdf`.
- [x] **Castillo numbers corrected to published Econometrica 93(5):1811–1854 (Sep 2025)** throughout schema, matrices, themes, gap analysis, §8, reference table, and Appendix A.
- [x] **Tresca venue corrected** to preprint arXiv 2504.06125, submitted to TCNS (under review, not in press).
- [x] **Sun 2024 venue confirmed**: TKDE 36(7), July 2024, KDD 2022 extension.
- [x] **Tag dictionary**: 59 tags partitioned across nine axes (v5 adds 8 tags + 2 axes).
- [x] **Synthesis matrices**: 6 matrices (Sections 3.1–3.6, with Matrix 6 added in v5 for the MFG-RL bridge).
- [x] **Themes**: 6 themes (2 consensus, 2 controversy [Theme 4 upgraded from v4's coverage gap], 2 gap).
- [x] **Gap justification**: Section 7 evaluates four candidate gaps with at least one counter-reason per gap; Gap A0 updated with published Castillo numbers and driver-heterogeneity calibration from Cook et al. 2021.
- [x] **Methodological caveats with citations**: Section 6.5 now cites Bojinov & Simchi-Levi 2023, Glynn-Johari-Rasouli 2020, and Hu & Wager 2022 (Appendix C) for switchback identification concerns.
- [x] **Forward link**: §8 connects synthesis to specific design choices; §7.1 connects gap analysis to a single scoped research question with v5-corrected numbers.
- [x] **Per-paper methodological notes carried**: Wang 2025, Hao 2023, Hu 2025 (from v4), plus v5 notes on Lyu 2025 and Tang 2019.
- [x] **Strategic-driver recommendation upgraded**: §8 recommendation now unconditional with empirical motivation from Hall-Horton-Knoepfle 2023 and Cook et al. 2021 (Thread G).

### 10.2 Self-attested items (reviewer judgment)

- [x] **MEAL paragraphs**: each theme paragraph in Section 5 has identifiable M, E, A, L.
- [x] **Summary-mode audit**: no paragraph in Section 5 is structured as "Author X said... Author Y said..."
- [x] **"Seminal" usage**: restricted to Gammelli 2021, Besbes 2021, Alonso-Mora 2017 (added in v5).
- [x] **Geographic/platform scope**: explicitly flagged in §1.1 and propagated into Theme 4, Gap A0, §8.
- [x] **Stance column convention**: position-taking, not reception-evaluation; each entry audited.

---

## 11. v6 expansion plan: what v5 still defers

v5 closed the v3/v4 corpus-size shortfall (now 38 unique papers, well within the 30–50 target) and addressed all of the prof's punch-list items at the manuscript level. The following items are deferred to v6:

1. **PDF re-verification of v5-added entries.** All 12 v5-added papers are currently `verified_against_abstract` only. v6 should add the PDFs to the working directory and re-verify each headline number against the source table/section, promoting entries to `verified_against_pdf` (or correcting them if the abstract-derived numbers differ from the PDF — analogous to the Castillo 2019→2025 correction in v5).
2. **Additional DRL pricing seed citations.** Mao et al. on dynamic-pricing RL; Tang et al.'s Didi pricing/incentive RL; Liu, Wu et al. on DRL surge (search returned several candidates but specific paper identification requires further triage). These remain the priority for testing whether Theme 4's controversy framing collapses (if the bridge literature is well-developed in DRL pricing too) or hardens (if the disconnect is real).
3. **Ride-pooling descendants of Alonso-Mora 2017.** Tachet et al. on shareability networks; Santi et al. on shared-mobility benefits; Simonetto et al. on real-time pooling.
4. **Auction-style / IC pricing papers beyond Ma-Fang-Parkes 2019.** Banerjee-Riquelme-Johari and follow-ons on dynamic pricing with strategic agents; descendants of the EC 2019 STP mechanism.
5. **Ride-hailing-specific MFG-RL applications.** No paper in v5's Thread F reports a ride-hailing evaluation; v6 should search for any 2024–2026 papers that take Yang 2018 / Guo 2019 / Cui-Koeppl 2021 and apply them specifically to ride-hailing dispatch or pricing.
6. **Driver-heterogeneity-as-agent papers.** v5's Theme 3 surfaces this as an absent third option. If any 2024–2026 paper has tried demographically-heterogeneous driver agents in MARL, v6 should find and integrate it.
7. **Replication / external validation of Lyu 2025 magnitudes.** The +48.87% upper end remains downweighted in v5; v6 should track whether any peer-reviewed follow-on or independent replication of LLM-as-agent dispatching produces magnitudes in the same range.

The single highest-value v6 addition is item 5 — a ride-hailing-specific MFG-RL application would close Theme 4's controversy at the empirical level, and Gap A0 would either be partly fulfilled or refined into a more specific question depending on the result.

---

## Appendix A: Per-claim `verified_against_pdf` and `verified_against_abstract` log

Verification provenance for every headline number. v3/v4 entries retain their `verified_against_pdf` status; v5-added entries are flagged `verified_against_abstract` until PDFs are added to the working directory.

| Claim | Paper / source | Source location | Verdict |
|---|---|---|---|
| **Castillo: +2.15% total welfare; +3.57% rider; −0.98% driver; −0.50% platform profits (all % of gross revenue), Houston** [v5 corrected] | Castillo, *Econometrica* 93(5):1811–1854 (Sep 2025) | Abstract; full text via Wiley DOI 10.3982/ECTA19106 | **`verified_against_abstract`** — published-version numbers verified against journal abstract; **supersedes v3/v4's working-paper-version numbers (+3.53%/+6.97%/−1.97%) which were roughly double the published magnitudes and did not split out platform profits**. v6 should re-verify against full PDF. |
| Yang 2024: ~5–6% IORR/IGMV (City B OPE) | `Yang2024_Rethinking_Order_Dispatching_KDD.pdf` | Table 1, Section 2.2 | `verified_against_pdf` (v3); framework name corrected from fabricated "FOVIR" to ODCMG |
| Enders 2023: 5% avg / 17% peak over 20 test dates | `Enders2023_Hybrid_MADRL_AMoD_L4DC.pdf` | Section 5 | `verified_against_pdf` (v3) with decomposition correction |
| Hoppe 2024: 1.9% vs LRA / 3.5% avg vs greedy / up to 6% peak | `Hoppe2024_GlobalRewards_MADRL_AMoD_L4DC.pdf` | Table 1, Section 4.1 | `verified_against_pdf` (v3) with attribution correction |
| Zhang 2023: average +6.4% / −10.6%; max +17% / −14% | `Zhang2023_FutureAware_Pricing_Matching_AAAI.pdf` | Abstract; Results | `verified_against_pdf` (v3) |
| Tresca 2025: NY 16-station ≈ −4.2%; Lux meso ≈ −2.4%; meta-RL ≈ +61% on Brooklyn special-event | `Tresca2026_RoboTaxi_Scale_TCNS.pdf` (= arXiv 2504.06125) | Table I, Table V, Section VI-D | `verified_against_pdf` (v3); v5 reframes venue as preprint, not in press |
| GARLIC: 38.23% empty-load vs. 40.71% best baseline = 2.48 pp | `Han2025_GARLIC_AAAI.pdf` | Table 1 | `verified_against_pdf` (v3) |
| Xu 2018: 0.5–5% revenue (RL-attributable) | `Xu2018_LargeScale_OrderDispatch_Didi_KDD.pdf` | Abstract | `verified_against_pdf` (v3) |
| **Tang 2019 (CVNet): qualitative gains in Didi A/B; transfer learning improves cross-city adaptability** [v5 added] | arXiv 2106.04493 / KDD 2019 proceedings | abstract & KDD acceptance summary | `verified_against_abstract` (v5); specific magnitudes pending PDF re-verification |
| Eshkevari 2022: 1.3% / 5.3% | `Eshkevari2022_RLinTheWild_Didi_KDD.pdf` | Abstract / experiments | `verified_against_pdf` (v3) |
| Yue 2024: 0.7–3.9% TDI / 1–2% CR | `Yue2024_End2End_Dispatch_Didi_CIKM.pdf` | Table 4 | `verified_against_pdf` (v3) |
| Zhang 2024 NondBREM: 3.76% ORR | `Zhang2024_NondBREM_OfflineRL_AAAI.pdf` | Table 2 | `verified_against_pdf` (v3) |
| Azagirre 2024 (Lyft): ≥$30M/yr | `Azagirre2024_Lyft_RL_Match.pdf` | Abstract | `verified_against_pdf` (v3) |
| Gammelli 2021: 1.6–2.2% from oracle | `Gammelli2021_GNN_RL_AMoD_CDC.pdf` | Tables I–II | `verified_against_pdf` (v3) |
| Gammelli 2023: 86.7%–99.5% of oracle | `Gammelli2023_GraphRL_BiLevel_ICML.pdf` | Tables 1–3 | `verified_against_pdf` (v3) |
| Singhal 2024: 89% (5-region NY); 100× speedup; 55–89% across scales | `Singhal2024_Electric_AMoD_GraphRL_ECC.pdf` | Table I, Figure 3 | `verified_against_pdf` (v3) |
| UniST: MAE 26.84 zero-shot, +10.1% | `Yuan2024_UniST_KDD.pdf` | Table 4 | `verified_against_pdf` (v3) |
| UrbanGPT: MAE 6.16 NYC, ~+28% | `Li2024_UrbanGPT_KDD.pdf` | Table 1 | `verified_against_pdf` (v3) |
| LLM-ODDR: paper-claimed +5.21–48.87% GMV; 1–5 s/decision | `Lyu2025_LLM-ODDR.pdf` | Table I; Section V.E | `verified_against_pdf` (v3) — numbers in source table; +48.87% upper end downweighted in v4/v5 cross-paper comparison |
| JDRCL: max-min fairness, convergence | `Sun2024_JDRCL.pdf` | Theorem 2; Appendix A | `verified_against_pdf` (v3); **v5 venue confirmed: TKDE 36(7), July 2024, KDD 2022 extension** |
| Wang 2025 (CoopRide): up to 12.4% over reported baselines | source PDF | Tables/results | `verified_against_pdf` (v3); v4 methodological note: upper-tail across baseline–dataset combinations |
| Hao 2023 (GAT-MF): 42.7% / −86.4% / −19.2% | source PDF | Tables/results | `verified_against_pdf` (v3); v4 methodological note: cross-task aggregate |
| Hu 2025 (BMG-Q): >50% overestimation reduction | source PDF | Tables/results | `verified_against_pdf` (v3); v4 methodological note: within-paper comparative against independent DQN |
| **Alonso-Mora 2017: 98% NYC demand served at 2.8 min mean wait, 3.5 min trip delay (cap 10)** [v5 added] | PNAS 114(3):462–467 | Abstract; Figure 2; Table 1 | `verified_against_abstract` (v5); v6 should re-verify against full PNAS PDF |
| **Ma-Fang-Parkes 2019: STP welfare-optimal in SPE; DSE impossibility** [v5 added] | arXiv 1801.04015 / EC 2019 | Theorem 1 (STP welfare); Theorem 2 (DSE impossibility) per abstract | `verified_against_abstract` (v5) |
| **Yang 2018 (Mean Field MARL): convergence to Nash on three benchmarks** [v5 added] | arXiv 1802.05438 / ICML 2018 | Abstract; experimental section | `verified_against_abstract` (v5) |
| **Guo 2019 (GMFG): unique-Nash existence; GMF-Q convergence** [v5 added] | arXiv 1901.09585 / NeurIPS 2019 | Abstract; Theorems 1–3 per abstract | `verified_against_abstract` (v5) |
| **Cui-Koeppl 2021: entropy-regularized convergence in non-contractive MFGs** [v5 added] | arXiv 2102.01585 / AISTATS 2021 | Abstract; main convergence theorem | `verified_against_abstract` (v5) |
| **Saldi-Başar-Raginsky: discrete-time risk-sensitive MFG existence + finite-horizon approximation** [v5 added] | published version | Existence theorem | `verified_against_abstract` (v5) |
| **Hall-Horton-Knoepfle 2023: hourly earnings re-equilibrate ~2 months after fare change** [v5 added] | NBER w30883 | Abstract; main empirical results | `verified_against_abstract` (v5); v6 should re-verify against full NBER PDF |
| **Cook et al. 2021: 7% gender earnings gap, fully decomposed into experience, location, driving speed** [v5 added] | *RES* 88(5):2210–2238 | Abstract; main decomposition | `verified_against_abstract` (v5) |
| **STGCN: outperforms FC-LSTM/DCRNN on multi-scale traffic** [v5 added] | arXiv 1709.04875 / IJCAI 2018 | Tables in Section 4 per abstract | `verified_against_abstract` (v5) |
| **DCRNN: SOTA multi-step traffic forecasting; advantage grows with horizon** [v5 added] | arXiv 1707.01926 / ICLR 2018 | Tables in main paper per abstract | `verified_against_abstract` (v5) |
| **Graph WaveNet: SOTA on PEMS-BAY/METR-LA via adaptive learned graph + dilated conv** [v5 added] | arXiv 1906.00121 / IJCAI 2019 | Tables in main paper per abstract | `verified_against_abstract` (v5) |

---

## Appendix B: Correction history (v1 → v5)

### B.1 Corrections inherited from v1

`Xu2018` (0.5–5% revenue, not 16–19%); `Eshkevari2022` (1.3% A/B, up to 5.3%, not 7–9%); `Yue2024` (0.7–3.9% TDI / 1–2% CR, not 8–12%); `Zhang2024 NondBREM` (3.76% ORR, not 10–15%); `Azagirre2024` ($30M/yr, not 5–8% acceptance); `Hao2023 GAT-MF` (42.7% / 86.4% / 19.2% as separate cross-task numbers, not a single ride-hailing dispatch number).

### B.2 Corrections inherited from v2

`Castillo` direction inverted in v1 (surge **increases** total welfare); `Yang2024` numbers re-sourced from City B OPE (~5–6%).

### B.3 Corrections introduced in v3

Enders 2023 / Hoppe 2024 / Zhang 2023 numbers re-decomposed; Yang 2024 framework name corrected from fabricated "FOVIR" to ODCMG; Tresca numbers rewritten; GARLIC arithmetic reconciled; Castillo working-paper citation hygiene flag added.

### B.4 Changes introduced in v4

Manuscript opening cleaned up; Castillo / Sun / Tresca venue framings adjusted; §6.5 parenthetical citations removed pending proper reference adds; Lyu 2025 +48.87% read skeptically; per-paper methodological notes added (Wang, Hao, Hu); Gap A0 re-staged honestly as structural-alignment exercise; §8 strategic-driver softened to conditional; §10 split into externally-verifiable / self-attested; "27 papers" replaced with cross-listing convention.

### B.5 Changes introduced in v5

- **Castillo numbers corrected to published Econometrica 93(5):1811–1854 (Sep 2025).** v3/v4 cited the 2019 working-paper version (+3.53%/+6.97%/−1.97%); the published version is +2.15% total welfare; +3.57% rider; −0.98% driver; −0.50% platform profits (all as % of gross revenue) — about half the magnitude and with platform profits separately reported. Updated throughout the manuscript (Theme 4, Gap A0, §8, Matrix 4, schema, reference table, Appendix A). The published version also adds the finding that female and long-hours drivers are most-hurt by surge — connected to Theme 3's new driver-heterogeneity-as-agent thread and to Gap A0's calibration.
- **Tresca venue corrected.** v3/v4 cited as "TCNS 2026 in press"; verified now as arXiv 2504.06125, submitted to TCNS, **under review (not in press)**. Citation reframed to preprint.
- **Sun 2024 venue confirmed.** TKDE 36(7), July 2024, KDD 2022 extension. The "venue confirmation pending" hedge from v3/v4 is dropped.
- **Thread F added (MFG-RL bridge): 4 papers** — Yang 2018, Guo 2019, Cui-Koeppl 2021, Saldi-Başar-Raginsky. Theme 4 upgraded from "coverage gap in this corpus" to a substantive controversy: the OR/economics pricing literature spans three traditions (equilibrium, mechanism design, structural empirical), and the MFG-RL bridge has built theoretical foundations, but the single DRL pricing paper has engaged with none of them.
- **Thread G added (driver labor): 2 papers** — Hall-Horton-Knoepfle 2023, Cook et al. 2021. §8's strategic-driver recommendation upgraded from conditional to unconditional. Theme 3 adds driver-heterogeneity-as-agent as a third absent option.
- **Thread H added (ST-GNN forecasting backbone): 3 papers** — STGCN, DCRNN, Graph WaveNet. Theme 2's encoder-vs-attention debate is now characterizable rather than only flagged.
- **Thread A bridge added.** Tang 2019 (CVNet KDD 2019, arXiv 2106.04493) added between Xu 2018 and Eshkevari 2022.
- **Thread D mechanism-design paper added.** Ma-Fang-Parkes 2019 STP mechanism (EC 2019, arXiv 1801.04015) added; Matrix 4 now spans four pricing traditions instead of three.
- **Thread B ride-pooling OR foundation added.** Alonso-Mora 2017 (PNAS 114(3):462–467) added; Hu 2025 BMG-Q comparison no longer un-moored on the OR side.
- **§6.5 properly cited.** Bojinov & Simchi-Levi 2023 (arXiv 2009.00148), Glynn-Johari-Rasouli 2020 (arXiv 2006.05591), Hu & Wager 2022 (arXiv 2209.00197) added as Appendix C methodological references; the four caveats now cite this literature rather than being stated as the reviewer's concerns.
- **Corpus size**: 38 unique papers (39 with cross-listing), comfortably within the guide's 30–50 target. The v3/v4 acknowledged shortfall is closed.
- **Tag dictionary expanded** to 59 tags across nine axes (added: MFG-RL axis, ST-GNN forecasting axis, driver-labor tag, mechanism-design tags, transfer-learning tag, foundational tag).
- **Matrix 6 added** for the MFG-RL bridge thread.
- **Verification provenance distinction added.** v5-added entries carry `verified_against_abstract`; v3/v4 entries retain `verified_against_pdf`. v6 should promote v5 entries after PDF re-checks.

---

## Appendix C: Methodological references for §6.5 (not corpus papers; cited for switchback identification under interference)

These references support the §6.5 caveats on switchback identification under spatial spillovers and temporal carryover. They are not part of the main corpus (which is ride-hailing-substantive); they are part of the methodological apparatus the reviewer uses to read the corpus's deployed-RL claims (Azagirre 2024 in particular).

1. **Bojinov, I. & Simchi-Levi, D. (2023).** "Design and Analysis of Switchback Experiments." *Management Science*, 69(7):3759–3777. arXiv 2009.00148. Establishes design-based causal-inference framework for switchback experiments under temporal carryover; documents the carryover-bias problem and analytical estimators.
2. **Glynn, P., Johari, R. & Rasouli, M. (2020).** "Adaptive Experimental Design with Temporal Interference: A Maximum Likelihood Approach." arXiv 2006.05591. Models switchback as a Markov-chain steady-state-difference estimation problem; characterizes optimal design and estimator under temporal interference.
3. **Hu, Y. & Wager, S. (2022).** "Switchback Experiments under Geometric Mixing." arXiv 2209.00197. Shows that standard switchback designs in geometrically-mixing Markov systems suffer estimation error decaying at T^{-1/3} rather than T^{-1/2}; burn-in periods recover close-to-optimal rates.

These three references jointly characterize what is known about the identification risk in production switchback experiments; the Lyft headline number (Azagirre 2024 ≥$30M/yr) should be read against this literature rather than as a clean causal estimate.

---

*End of literature review (v5).*
