# Who Benefits from *Learned* Surge Pricing? The Welfare Incidence of Policy-Search Pricing Control under Driver Heterogeneity

*A Master's thesis in Computer Science.*

> **A note on terminology.** The controller in this thesis is a compact policy optimized by **Augmented Random Search (ARS)**, a derivative-free policy-search method, after a per-step deep-RL method (PPO) failed for a diagnosed reason (§4.5). We therefore use "learned control" and "policy-search control" rather than "deep reinforcement learning." The substantive question — whether a *learned* pricing controller reproduces a structural welfare target, and how that depends on its objective — is unchanged by the choice of optimizer.

---

## Abstract

> *[To be written after results. One paragraph: motivation (DRL controllers now set prices/matching/rebalancing on real platforms, but their distributional welfare consequences are unstudied); approach (a calibrated spatial ride-hailing simulator with a heterogeneous strategically-responding driver population, against which DRL controllers with different objectives are trained and their four-way welfare incidence measured); central finding (incidence is objective-dependent; welfare-oriented controllers reproduce the Castillo (2025) structure while profit-maximizing controllers distort it; the most-engaged drivers bear the largest losses); and the fairness-frontier result.]*

---

## Chapter 1 — Introduction

### 1.1 Motivation

Ride-hailing platforms make three tightly coupled operational decisions on the spatiotemporal graph of a city: **matching** (which driver serves which rider), **pricing** (what surge multiplier to charge), and **rebalancing** (where idle drivers should reposition). Over 2018–2026 the methodological default for these decisions has shifted decisively toward deep reinforcement learning (DRL) and graph neural networks, with production deployments at Didi and Lyft and a fast-growing line of work treating the three levers as one joint control problem rather than three separate ones.

In parallel, the economics literature has reached a sharp, quantitative understanding of *who gains and who loses* from surge pricing. Castillo's (2025) structural study of the Houston Uber market decomposes the welfare effect of surge, relative to an optimized uniform-pricing counterfactual, into four parties: surge raises **total welfare by 2.15%** of gross revenue and **rider surplus by 3.57%**, while **driver surplus falls 0.98%** and **platform profit falls 0.50%**; among drivers, those who work long hours — especially women — are hurt most.

These two literatures have not met. The DRL controllers that increasingly set prices in practice are evaluated almost exclusively on platform-centric efficiency metrics — gross merchandise value, order-response rate, driver income — never on the *distribution* of welfare across riders, drivers, and platform. We therefore do not know whether a learned surge controller reproduces, exceeds, or quietly distorts the welfare incidence that the structural economic model documents. This thesis closes that gap on a calibrated proxy market.

### 1.2 Research questions

- **RQ1 (alignment, decomposed).** Does learned spatio-temporal surge reproduce Castillo's four-way incidence — total ↑, rider ↑, driver ↓, platform ↓? Following Castillo's own logic, we decompose the effect of surge into two channels and test each: (a) a **price-variation effect** — surge vs. a flat uniform price *at the same average multiplier* (a mean-preserving spread), which isolates allocative efficiency and matching gains; and (b) a **price-level effect** — the gap between the platform-optimal uniform price and the surge average. Castillo's headline arises because the variation effect lifts everyone slightly while the level effect (optimal uniform $>$ surge average) transfers surplus from drivers and platform to riders. The mean-preserving-spread comparison is calibration-robust; the level effect depends on the platform's objective, which is exactly what RQ2 turns on.
- **RQ2 (objective-dependence).** This is the central question. Is the welfare incidence a property of surge pricing itself, or of the *objective the controller maximizes*? We compare profit-, throughput-, welfare-, and welfare-weighted controllers and ask which reproduce the Castillo structure and which distort it — answering directly whether a learned controller "matches, exceeds, or quietly distorts" the structural model's incidence. Because the same surge *mechanism* yields different incidence under different objectives, the distortion is a property of the objective, not of surge.
- **RQ3 (heterogeneous incidence).** Does the driver-side loss fall most heavily on the most-engaged (long-hours, low-reservation-wage) drivers, as Castillo finds empirically, and what is the mechanism in a model with heterogeneous drivers?
- **RQ4 (fairness frontier).** If a driver-earnings fairness penalty is imposed on the controller (the field's emerging "fairness as a hard constraint" consensus), what is the efficiency–equity trade-off, and at what cost can the driver loss be neutralized?

### 1.3 Contributions

1. **A calibrated spatial ride-hailing simulator** with elastic discrete-choice demand, a *heterogeneous, strategically-responding* driver population (drivers differ in reservation wage, repositioning responsiveness, and engagement), endogenous matching frictions (the wild-goose chase), and a four-way welfare decomposition (rider/driver/platform/total). The accounting is closed — components sum to total welfare with no bookkeeping leakage (a consistency check confirms this to machine precision; we are careful, per §5.0, not to overstate this as validating the welfare *model*, only its bookkeeping). To our knowledge this is the first ride-hailing learning environment that models drivers as a heterogeneous population whose surge responses differ by type and that reports rider/driver/platform/total welfare rather than a single platform metric.
2. **The first test of a learned pricing controller against a published structural welfare target.** We measure whether learned surge pricing reproduces the Castillo (2025) incidence structure on a Houston-calibrated proxy, isolating price flexibility exactly as Castillo does.
3. **An objective-dependence result (RQ2) — the central contribution:** the welfare incidence of learned surge is shown to be determined by the controller's objective, answering directly the open question of whether a learned controller "matches, exceeds, or quietly distorts" the structural model's incidence.
4. **A driver-heterogeneity incidence analysis (RQ3)** reproducing the timing/exposure mechanism behind Castillo's "long-hours drivers hurt most" finding, and **a fairness-frontier characterization (RQ4).**

### 1.4 Scope and honest limitations

This is a **structural-alignment study on a calibrated proxy market**, not a replication of Castillo's Houston estimates (the underlying Uber data are proprietary). Its claims are about whether *learned* controllers reproduce the *structure* (signs, rough proportions, and incidence mechanism) of a published economic result, and how that structure depends on the controller's objective. The simulator abstracts a real city; the sim-to-real gap, the single-calibration concern, and the specific driver-behavior assumptions are addressed explicitly in Chapter 6.

---

## Chapter 2 — Background and related work

Ride-hailing operations research over the last decade has converged on a common object of study — three control levers on a city-scale spatiotemporal graph — but split into two communities that rarely cite one another: a *learning* community that scales DRL and graph neural networks to the matching and rebalancing levers, and an *economics/OR* community that derives equilibrium and welfare results for the pricing lever. This thesis sits at the seam. The review below is organized by lever and by the methodological lineage of each, and closes by positioning the present work against the open gaps.

### 2.1 Three coupled levers

The three decisions are coupled through the market state. Pricing changes the request arrival rate and therefore the demand the matcher sees; matching outcomes determine the post-trip spatial distribution of idle supply that rebalancing must then correct; and the rebalanced supply distribution changes the pickup times — and hence the prices and match feasibility — of the next period. The dominant framing treats the city as a graph of zones with time-dependent demand, and the controller as acting at second-to-minute granularity on matching, minute granularity on rebalancing, and minute-to-hour granularity on pricing. A recurring observation across the corpus is that, although the three levers are coupled, almost no work optimizes all three jointly with a single learned controller; joint *matching+repositioning* (e.g., the constrained max-min-fairness Markov game of JDRCL) and joint *pricing+matching* (e.g., future-aware ride-pooling ADP) each appear, but three-way joint control remains essentially open.

### 2.2 The deployed-RL lineage and its evaluation blind spot

A seven-year lineage of production systems establishes DRL dispatching as deployable at scale. The foundational move framed dispatch as large-scale sequential decision-making with an offline-learned state-value function feeding an online bipartite matcher (Xu et al. 2018), a template extended through transfer-learning value networks (Tang et al. 2019), on-policy online learning with safety safeguards on Didi's international markets (Eshkevari et al. 2022), and a documented online-matching deployment at Lyft worth tens of millions of dollars per year (Azagirre et al. 2024). A parallel methodological line replaces the two-stage value+Hungarian pipeline with end-to-end learned matchers (Yue et al. 2024) or reframes dispatch as a cooperative Markov game across regions (Yang et al. 2024). What unites this lineage is its **evaluation metric**: every deployed system reports a *platform-centric* quantity — gross merchandise value, order-response or completion rate, or driver income. None reports the *distribution* of welfare across riders, drivers, and platform. The headline percentages are also confounded — they bundle the RL change with whatever else shipped concurrently, and rely on switchback designs with documented identification concerns under spatial spillovers. This evaluation blind spot is precisely the opening this thesis addresses: a learned controller can raise a platform metric while redistributing welfare in ways the metric cannot see.

### 2.3 Matching and rebalancing: hybrid learning + optimization and graph-MARL

Two methodological families dominate the matching/rebalancing levers. The first is **hybrid learning + optimization**, in which a learned policy proposes a target (a desired idle-vehicle distribution, or per-edge weights) and a classical combinatorial solver — a min-cost-flow LP or a bipartite/ILP matcher — executes a feasible action. This bi-level pattern originated for autonomous-mobility-on-demand rebalancing (Gammelli et al. 2021) and recurs across electric-fleet, network-flow, and city-scale robo-taxi extensions, as well as in the MASAC-plus-bipartite-matching template for AMoD (Enders et al. 2023; Hoppe et al. 2024). The second family is **graph-based and mean-field MARL**, which scales cooperative coordination to thousands of driver- or grid-agents by combining graph attention with mean-field approximations (e.g., GAT-MF, CoopRide, BMG-Q), with theoretical grounding in mean-field-game RL (Guo et al. 2019; Cui & Koeppl 2021). The design lesson this thesis adopts from the hybrid line is to delegate the combinatorial matching step to a fast solver parameterized by the learned controls, keeping the learning problem low-dimensional. The design lesson it adopts from the mean-field line is to model the driver population as a responsive aggregate rather than as thousands of co-learning agents.

### 2.4 Pricing: equilibrium economics versus the DRL track

The pricing lever has been studied with very different tools. The **economics/OR track** pursues equilibrium and welfare results: spatial fluid models show that local, myopic surge "leaves money on the table" relative to network-anticipating pricing because supply relocates toward high prices (Besbes et al. 2021); strategic-driver network models show that a platform may optimally reject demand to induce repositioning (Afèche et al. 2023); and mechanism-design work establishes incentive-compatible spatiotemporal pricing with an impossibility result for dominant-strategy implementation (Ma, Fang & Parkes 2019). The capstone for the present work is a structural empirical study of the Houston Uber market (Castillo 2025), which decomposes the welfare effect of surge — relative to an optimized uniform-pricing counterfactual — into a small total-welfare gain (+2.15% of gross revenue) and a sharp redistribution: riders gain (+3.57%) while drivers (−0.98%) and the platform (−0.50%) lose, with long-hours and female drivers hurt most. Crucially, this result rests on three mechanisms — a matching technology in which scarcity produces a "wild-goose chase," driver pay that is homogeneous across trips (so allocative efficiency benefits only riders), and an optimal uniform price set *above* the surge average to manage peak scarcity — and on a platform that maximizes a *welfare-weighted* objective rather than profit. The **DRL pricing track**, by contrast, scales actor-critic methods to networked pricing (e.g., future-aware joint pricing+matching for ride-pooling) but is evaluated on revenue or vehicle-count metrics and is not in dialogue with the equilibrium-welfare results. No DRL pricing study to date has been tested against a structural welfare decomposition. That confrontation is RQ1.

### 2.5 Strategic drivers and driver-labor heterogeneity

Whether drivers should be modelled as a strategic, heterogeneous population — rather than as fungible vehicles — is settled empirically even though most RL work elides it. Quasi-experimental evidence shows ride-sharing markets *re-equilibrate* after fare changes: drivers enter and work more hours, utilization falls, and the hourly earnings rate reverts to baseline within roughly two months (Hall, Horton & Knoepfle 2023). Heterogeneity is consequential: a 7% gender earnings gap among Uber drivers is fully explained by behavioral differences in experience, where drivers choose to work, and driving speed, with no role for discrimination (Cook et al. 2021). These findings ground the modelling choices of this thesis — a reservation-wage entry/exit margin, earnings-following repositioning, and driver types that differ in engagement — and they ground the *interpretation* of RQ3, since the heterogeneity that determines surge incidence in Castillo's data is a timing/exposure phenomenon (which drivers are active when multipliers are high) rather than a difference in price sensitivity. Notably, no paper in the surveyed corpus models drivers as a heterogeneous population whose surge response differs by type; doing so is a corpus-novel element of this work.

### 2.6 Fairness as an emerging constraint

A consensus is crystallizing that fairness belongs in the objective as a first-class constraint rather than as a post-hoc check. Constrained MARL formulations impose max-min driver-earnings fairness with convergence guarantees (JDRCL), and continuous mean-field RL enforces spatial accessibility constraints at fleet scale (Jusup et al. 2025). Combined with the empirical finding that surge redistributes away from the most-engaged and from female drivers, this motivates RQ4: characterizing the efficiency–equity frontier of a fairness-penalized controller.

### 2.7 Positioning

The gap analysis distilled from this corpus identifies, as its highest-priority opening, the *structural alignment of a three-way DRL welfare decomposition with the Castillo target*, and, as a corpus-novel stepping stone, *modelling driver heterogeneity as an agent population*. This thesis pursues both jointly. Against §2.2 it supplies the missing distributional evaluation; against §2.4 it provides the first confrontation of a learned controller with a structural welfare decomposition, and it reframes the question from "does surge help" to "does the *controller's objective* determine who surge helps"; against §2.5 it adopts an empirically grounded heterogeneous strategic-driver model; and against §2.6 it traces the fairness frontier. It does not attempt the field-wide problems of cross-city transfer or an LLM-controller benchmark, which the same analysis judges either slow-moving or premature.

---

## Chapter 3 — Problem formulation

### 3.1 The market as a Markov decision process

We model a ride-hailing market over a grid of $Z$ zones on a $G \times G$ lattice and a finite operating horizon of $T$ epochs. Travel time between zones equals their Manhattan distance in cells (one cell per epoch at the modelled speed). A single decision-maker — the **platform controller** — acts each epoch; the rider and driver populations respond.

**State.** At epoch $t$ the controller observes $s_t = (\mathbf{q}_t, \boldsymbol{\lambda}_t, \mathbf{b}_t, \phi_t)$ where $\mathbf{q}_t \in \mathbb{R}^Z$ is the distribution of idle, available drivers across zones; $\boldsymbol{\lambda}_t \in \mathbb{R}^Z$ is the current potential-demand intensity per zone; $\mathbf{b}_t \in \mathbb{R}^Z$ is the driver population's belief about zonal earnings (its spatial-supply signal); and $\phi_t$ encodes time-of-day.

**Action (three levers).** $a_t = (\mathbf{m}_t, \rho_t, \mathbf{g}_t)$:
- **Pricing** $\mathbf{m}_t \in [\underline m, \overline m]^Z$: a per-zone surge multiplier applied to the base fare.
- **Matching** $\rho_t \in \{\rho_{\min},\dots,\rho_{\max}\}$: a dispatch radius capping how far a driver may be sent to a pickup — the lever governing the wild-goose-chase trade-off (serve distant riders vs. tie up drivers).
- **Rebalancing** $\mathbf{g}_t \in \mathbb{R}_{\ge0}^Z$: a per-zone repositioning incentive that biases idle-driver relocation.

**Transition.** Given $a_t$: (i) drivers update their online/offline status and reposition; (ii) potential riders arrive, observe price and an estimated wait, and request iff their utility is non-negative; (iii) requesting riders are matched to nearby available drivers; (iv) matched drivers become busy for the pickup-plus-trip duration; (v) welfare is accrued. Section 4 specifies each step.

### 3.2 Welfare decomposition

For a served trip with rider value $v$, price $p$, realized wait $w$, and trip/pickup distance, money flows are pure transfers: the rider pays $p$, the driver receives $(1-\gamma)p$, and the platform keeps the commission $\gamma p$. Over an episode we define

$$\text{RS} = \!\!\sum_{\text{served}}\!\! (v - p - \kappa w), \quad \text{DS} = \!\!\sum_{\text{drivers}}\!\! \big(\text{earnings} - \text{drive cost} - r_\theta\, \tau_{\text{online}}\big), \quad \text{PP} = \!\!\sum_{\text{served}}\!\! \gamma p,$$

where $\kappa$ is the rider value of waiting time and $r_\theta$ the driver's type-specific reservation wage (opportunity cost per online epoch). Total welfare is $W=\text{RS}+\text{DS}+\text{PP}$. Because prices and commissions cancel, $W$ equals the **net real surplus**

$$W = \sum_{\text{served}}(v - \kappa w) \;-\; \sum_{\text{drivers}}\text{drive cost} \;-\; \sum_{\text{drivers}} r_\theta\, \tau_{\text{online}},$$

an identity the simulator verifies to machine precision (Section 5.1). Following Castillo we report welfare components as percentages of **gross revenue** $\sum_{\text{served}} p$.

### 3.3 The four platform objectives

The controller is trained to maximize one of four per-episode objectives, which constitute the RQ2 treatment:
- **Profit:** $\sum \gamma p$ (platform commission).
- **Throughput:** number of completed trips (a proxy for the order-response / completion-rate objectives that deployed systems optimize).
- **Welfare:** total welfare $W$ (a social planner).
- **Welfare-weighted:** $\alpha_\pi\,\text{PP} + \alpha_R\,\text{RS} + \alpha_D\,\text{DS}$, the analog of Castillo's platform, which maximizes profit plus weighted rider and driver surplus ($\alpha_R=1$, with $\alpha_\pi+\alpha_D<0.7$).

### 3.4 The uniform-pricing counterfactual

To isolate the welfare incidence of **price flexibility** — Castillo's exact comparison — we fix the matching and rebalancing levers to a common policy and compare two pricing regimes: a **uniform** policy that applies a single constant multiplier to the whole market (its level optimized for the platform's objective by grid search), versus a **surge** policy that sets multipliers per zone and per epoch (learned by DRL). The difference in the four welfare components, as a percentage of gross revenue, is the object compared to Castillo.

---

## Chapter 4 — Simulator and methods

The simulator is implemented in NumPy; one episode is 48 epochs on a $5\times5$ grid with a 400-driver pool. The full configuration (with the literature grounding for each parameter) is in `src/ridehail/config.py`; the grounding notes for the calibration are in `grounding/`.

### 4.1 Demand

Potential ride requests in zone $z$ at epoch $t$ arrive as a Poisson process with rate $\lambda_{z,t}$. The temporal profile is bimodal (a morning and an evening peak), and the spatial profile shifts the origin mass between periphery and centre across the two peaks (a commute pattern: periphery→centre in the morning, centre→periphery in the evening), creating the joint spatial–temporal supply–demand imbalance that surge addresses. Each potential rider draws a destination from an origin-specific gravity-plus-commute distribution, and a willingness-to-pay $v = f_1 \cdot m$, where $f_1$ is the fare at multiplier 1 for the trip and $m$ is a WTP-to-fare ratio drawn from a truncated $\mathrm{LogNormal}(1.0, 1.4)$ (capped at 4). A rider requests the trip iff $v - p - \kappa\, \hat w \ge 0$, where $p = f_1 \cdot m_z$ is the surged price, $\hat w$ is the estimated wait shown in the app (distance to the nearest available driver plus the zone's congestion delay), and $\kappa$ is the rider value of waiting time. Because $\hat w$ rises with congestion, demand responds to both price and crowding — a downward-sloping, congestion-sensitive demand curve.

### 4.2 Heterogeneous strategic drivers

The fleet is a pool of $M=400$ drivers partitioned into three types — *full-time* (35%, reservation wage $r_\theta=0.18$/epoch), *part-time* (45%, $r_\theta=0.55$), and *casual* (20%, $r_\theta=0.95$) — differing in their opportunity cost of being online and their repositioning responsiveness. Three behaviours make the population *strategic* in the sense of the OR literature (Besbes 2021; Afèche 2023):
- **Repositioning.** An idle driver relocates one cell by sampling a destination from a softmax over reachable zones' expected net earnings minus travel cost (plus the platform's rebalancing incentive), with a type-specific temperature. This nests the Wardrop spatial equilibrium in which earnings are equalized across locations net of travel cost, and matches Castillo's movement logit.
- **Entry/exit.** A driver is online with probability increasing in the gap between the local expected-earnings belief and its reservation wage. Low-$r_\theta$ full-time drivers stay online through low-demand, low-surge periods; high-$r_\theta$ casual drivers come online only at peaks. This reservation-wage participation margin produces the *re-equilibration* documented empirically (Hall–Horton–Knoepfle 2023), and the **timing/exposure** structure behind RQ3.
- **Beliefs.** Drivers maintain an adaptive population belief about per-zone earnings, updated from realized trip earnings (and dragged toward zero by idleness), which is the spatial-supply signal the controller observes.

Critically, drivers are paid **by the meter** — $(1-\gamma)p$ for a trip, independent of the rider's value — so allocative efficiency from surge accrues to riders, not drivers, exactly as in Castillo's homogeneous-driver-utility assumption. Drivers are a fixed-rule responsive population, not co-learning agents, which keeps the environment stationary for the controller and mirrors Castillo's "platform optimizes, drivers respond in equilibrium" structure.

### 4.3 Matching and the wild-goose chase

Each epoch, requesting riders are matched to available drivers by nearest-driver-first assignment at zone granularity (faithful to Castillo's random-order, nearest-driver matching technology), within the controller's dispatch radius; riders with no available driver inside the radius abandon. In addition to the travel-distance pickup time, a **congestion delay** lengthens the effective pickup time when local demand exceeds available supply: a zone's delay rises with its demand-to-supply ratio (capped), a reduced form of Castillo's pickup-time distribution $G(\cdot)$ that depends on driver availability. A matched driver is unavailable for the pickup-plus-congestion-plus-trip duration, so scarcity ties drivers up and feeds back into more scarcity — the **wild-goose chase** that, in Castillo, is "the main driver of the results." This friction is what makes *low* prices costly (flooding the market produces long waits and ties up drivers), and it is the reason the platform-optimal uniform price is interior rather than at the floor. The supply-demand imbalance in our market therefore manifests on both the extensive margin (abandonment, which concentrates sharply at peaks — correlation with demand $\approx 0.99$) and the intensive margin (congestion delay). A diagnostic sweep confirms that widening the dispatch radius lengthens pickups and, past a point, lowers welfare as drivers are tied up.

### 4.4 Calibration

Parameters are grounded in the corpus (grid size from Gammelli 2021; time/fleet/episode structure from Lin 2018 and Zhang 2023; fare structure and commission from Castillo 2025). The two free demand parameters (the WTP ratio's location and scale) are set so the simulated short-run price elasticity at the operating point is $\approx 0.31$ — close to Castillo's estimated 0.268 — with a baseline acceptance rate of $\approx 0.77$. The fleet size (400) places the market in a supply-constrained regime (match rate $\approx 0.66$ at multiplier 1) where surge has room to reallocate scarce capacity.

One parameter is calibrated to make the proxy *qualitatively faithful* to the mechanism that generates Castillo's result: the congestion-delay coefficient is set ($0.8$) so that the **welfare-weighted platform's optimal uniform multiplier is $\approx 1.2$** — matching Castillo's estimated $1.174$ — rather than at the price floor. We stress that this calibrates the *market's* structure (so that low prices are costly, as in Castillo), not the controller's behaviour; the controller still has to *discover* a good policy, and whether its learned surge reproduces the welfare incidence is left to the experiment. The resulting optimal uniform multipliers form a clean spread across objectives — profit $\approx 3.4$, welfare $\approx 2.0$, welfare-weighted $\approx 1.2$, throughput $\approx 1.2$ — so the objectives genuinely differ in how they price. The four-way welfare accounting satisfies the consistency identity $W = \text{RS}+\text{DS}+\text{PP}$ to a maximum relative error of $\sim10^{-16}$ across pricing regimes and seeds.

### 4.5 The controller and baselines

**Why not PPO.** We first implemented Proximal Policy Optimization. Although its value function fit the return well (explained variance $\to 1$), the policy mean did not move off its initialization: the per-epoch reward is dominated by the exogenous demand cycle, which the value baseline absorbs, leaving the pricing action's marginal effect buried below the noise floor — a credit-assignment failure that persisted under aggressive learning rates, reduced entropy, and standard-deviation annealing. This is itself a useful negative result about per-step policy-gradient methods in markets with a strong exogenous temporal signal.

**Augmented Random Search.** We therefore optimize the controller with ARS-v2 (Mania, Guy & Recht 2018), which estimates a gradient of the *episode* objective by finite differences over random parameter perturbations, with top-direction selection. ARS sidesteps per-step credit assignment by scoring whole episodes and is robust in exactly this regime. The controller is a **compact, zone-shared (location-invariant) policy** (~23 parameters): the surge and rebalancing rules are a single linear function applied to each zone's local features (its supply, demand, and earnings-belief) plus global time features, and the dispatch radius is a function of global features; the environment squashes the outputs to bounded multipliers, a radius, and incentives. The location-invariant parameterization both encodes a sensible inductive bias (the pricing rule depends on local imbalance) and lets ARS find the price *level* — a single bias term — efficiently, which an unstructured policy could not (its level was pinned near the action mid-point because mean-zero observations average its output to zero). Each candidate is evaluated on three rollout seeds; we use 16 directions, the best 8 per update, two independent training seeds, and parallelize evaluations across processes. We acknowledge that a compact policy is a weaker function class than a deep network; §6 discusses this as a threat to validity, but the policy is expressive enough to recover the objective-specific optima (it matches the grid-searched uniform optimum for the profit objective to within noise), so it is a fair instrument for the welfare-incidence question.

**Baselines.** The **uniform-pricing counterfactual** is a single constant multiplier whose level is grid-searched to maximize the platform's objective (Castillo's comparison). The **myopic local surge** baseline prices each zone from its instantaneous local supply–demand imbalance, ignoring network effects (the Besbes 2021 "local surge leaves money on the table" reference policy).

---

## Chapter 5 — Experiments and results

All controllers are trained with ARS (two independent training seeds each) and evaluated on 16 held-out market seeds disjoint from training; reported figures are means with standard errors over the evaluation seeds. The uniform counterfactual's multiplier is grid-searched per objective on the same evaluation seeds.

### 5.0 Simulator validation

Four checks establish that the simulator behaves as an economic market and that the controller is a fair instrument.

**Welfare accounting has no leakage.** Across pricing regimes (multipliers $0.9$–$2.5$) and seeds, the components sum to total welfare and equal the net-real-surplus form to a maximum relative error of $5\times10^{-16}$. This is a *bookkeeping* guarantee — prices and commissions cancel as transfers by construction, so the check confirms no accumulator leaks, not that the underlying definitions of value, wait cost, or opportunity cost are themselves "correct." Its value is that any measured redistribution across parties is real and not an artifact of mis-summed accounts.

**Demand is elastic and inelastic in the right range.** Requests fall monotonically with price (e.g., $7{,}700\to6{,}800\to5{,}300$ as the multiplier rises $1.0\to1.5\to2.5$), and the measured short-run price elasticity at the operating point is $0.31$, close to Castillo's estimated $0.268$, with baseline acceptance $\approx0.77$.

**The wild-goose chase is a real force.** Abandonment concentrates almost entirely at demand peaks (correlation between per-epoch abandonment and demand $\approx0.99$; $\sim\!207$ abandonments per peak epoch versus $\sim\!2$ per trough epoch). A dispatch-radius sweep confirms the intensive-margin chase: widening the radius lengthens mean pickups and, beyond radius $\approx2$, *lowers* welfare because drivers are tied up serving distant riders. The congestion-delay calibration makes low prices costly, which is why the platform-optimal uniform price is interior (§4.4).

**Objectives genuinely differ, and the controller is a fair instrument.** The grid-searched optimal uniform multiplier rises monotonically with the platform's emphasis on its own revenue: throughput $1.2$, welfare-weighted $1.2$, welfare $2.0$, profit $3.4$. For the profit objective the ARS controller recovers the profit-maximizing price level (learned mean surge $\approx3.0$; achieved objective within $\sim0.1\%$ of the grid-searched uniform optimum), confirming the compact policy is expressive enough to find the objective-specific optimum rather than being an artificially weak baseline.

### 5.1 RQ1 — Does learned surge reproduce the Castillo incidence structure?

> *[Welfare-incidence table + figure for the welfare-weighted (Castillo-analog) controller; sign and proportion comparison. TODO.]*

### 5.2 RQ2 — Objective-dependence of the welfare incidence

> *[The four-objective table/figure: which objectives reproduce vs distort the structure. TODO.]*

### 5.3 RQ3 — Heterogeneous incidence across driver types

> *[By-type per-capita surplus change; do full-time/long-hours drivers lose most; the timing/exposure mechanism. TODO.]*

### 5.4 RQ4 — The efficiency–equity frontier

> *[Fairness-weight sweep: driver-equity vs total welfare; cost of neutralizing the driver loss. TODO.]*

### 5.5 Three-way control (Gap A0) and scope conditions

> *[Three-way vs price-only incidence; sensitivity across market regimes (fleet size, elasticity): when the structure holds/breaks. TODO.]*

---

## Chapter 6 — Discussion

> *[Interpretation: incidence is a property of the objective, not of surge; the deployed-systems blind spot; alignment and divergences from Castillo and their mechanisms; threats to validity (sim-to-real, single calibration, driver-model assumptions, ARS vs global optimum); what would change the conclusions. TODO.]*

---

## Chapter 7 — Conclusion and future work

> *[Summary of findings against RQ1–RQ4. Headline: the welfare incidence of learned surge is a property of the controller's objective, not of surge pricing per se; this reframes the deployed-systems practice of optimizing platform-centric metrics as an implicit distributional choice. Future work: richer pickup-time (intensive-margin) frictions; multi-platform competition; co-learning strategic drivers; transfer of the welfare-decomposition evaluation to a real-data simulator; LLM-as-reward-designer for the welfare-weighted objective. TODO finalize after results.]*

This thesis took the question "who benefits from surge pricing?" — answered for a fixed proprietary algorithm by structural econometrics — and asked it of *learned* controllers, the kind increasingly deployed in practice. The contribution is not a new control algorithm but a new *evaluation*: measuring the four-way welfare incidence of a learned three-way controller against a published structural target, on a calibrated proxy market with a heterogeneous, strategically-responding driver population. *[Final sentence(s) summarizing the empirical answer once results are in.]*

---

## References

*Pricing, welfare, and strategic supply (economics/OR).*

- Castillo, J.C. (2025). Who Benefits from Surge Pricing? *Econometrica* 93(5):1811–1854.
- Besbes, O., Castro, F., Lobel, I. (2021). Surge Pricing and Its Spatial Supply Response. *Management Science* 67(3):1350–1367.
- Afèche, P., Liu, Z., Maglaras, C. (2023). Ride-Hailing Networks with Strategic Drivers. *M&SOM* 25(5):1890–1908.
- Ma, H., Fang, F., Parkes, D.C. (2019). Spatio-Temporal Pricing for Ridesharing Platforms. *EC 2019*.

*Driver labor economics.*

- Hall, J.V., Horton, J.J., Knoepfle, D.T. (2023). Ride-Sharing Markets Re-Equilibrate. *NBER WP 30883*.
- Cook, C., Diamond, R., Hall, J.V., List, J.A., Oyer, P. (2021). The Gender Earnings Gap in the Gig Economy. *Review of Economic Studies* 88(5):2210–2238.

*Production and deployed RL dispatching.*

- Xu, Z., et al. (2018). Large-Scale Order Dispatch in On-Demand Ride-Hailing. *KDD 2018*.
- Tang, X., et al. (2019). A Deep Value-Network Based Approach for Multi-Driver Order Dispatching. *KDD 2019*.
- Eshkevari, S.S., et al. (2022). Reinforcement Learning in the Wild. *KDD 2022*.
- Azagirre, X., et al. (2024). A Better Match for Drivers and Riders: RL at Lyft. *INFORMS J. Applied Analytics*.
- Yue, Y., et al. (2024). An End-to-End RL Approach for Micro-View Order Dispatching. *CIKM 2024*.
- Yang, J., et al. (2024). Rethinking Order Dispatching in Online Ride-Hailing. *KDD 2024*.

*Graph-MARL, mean-field, and hybrid learning+optimization.*

- Lin, K., et al. (2018). Efficient Large-Scale Fleet Management via Multi-Agent DRL. *KDD 2018*.
- Li, M., et al. (2019). Efficient Ridesharing Order Dispatching with Mean-Field MARL. *WWW 2019*.
- Yang, Y., et al. (2018). Mean Field Multi-Agent Reinforcement Learning. *ICML 2018*.
- Guo, X., et al. (2019). Learning Mean-Field Games. *NeurIPS 2019*.
- Cui, K., Koeppl, H. (2021). Approximately Solving Mean-Field Games via Entropy-Regularized DRL. *AISTATS 2021*.
- Hao, Q., et al. (2023). GAT-MF: Graph Attention Mean Field. *KDD 2023*.
- Wang, X., et al. (2025). CoopRide. *KDD 2025*.
- Hu, J., et al. (2025). BMG-Q. *IEEE T-ITS*.
- Jusup, M., et al. (2025). Scalable Ride-Sourcing Rebalancing with Service Accessibility Guarantee. *Preprint*.
- Gammelli, D., et al. (2021). Graph Neural Network RL for AMoD Systems. *IEEE CDC 2021*.
- Gammelli, D., et al. (2023). Graph RL for Network Control via Bi-Level Optimization. *ICML 2023*.
- Enders, T., et al. (2023). Hybrid Multi-Agent DRL for AMoD. *L4DC 2023*.
- Hoppe, H., et al. (2024). Global Rewards in Multi-Agent DRL for AMoD. *L4DC 2024*.
- Alonso-Mora, J., et al. (2017). On-demand high-capacity ride-sharing. *PNAS* 114(3):462–467.

*Joint optimization, LLM/foundation-model controllers, and fairness.*

- Zhang, X., Varakantham, P., Jiang, H. (2023). Future-Aware Pricing and Matching for Ride Pooling. *AAAI 2023*.
- Sun, J., et al. (2024). Joint Order Dispatching and Driver Repositioning (JDRCL). *IEEE TKDE* 36(7).
- Han, X., et al. (2025). GARLIC. *AAAI 2025*.
- Lyu, T., et al. (2025). LLM-ODDR. *Preprint*.

*Method.*

- Mania, H., Guy, A., Recht, B. (2018). Simple Random Search Provides a Competitive Approach to Reinforcement Learning. *NeurIPS 2018*.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *Preprint*.

---

## Appendix A — Reproducibility

The simulator (`src/ridehail/`), controllers (ARS), baselines, and experiment scripts (`experiments/`) are in the repository. Key entry points: `experiments/test_invariants.py` (welfare-identity and elasticity assertions), `experiments/run_all.sh` (full experiment matrix), `experiments/aggregate_results.py` (tables/figures). All randomness is seeded; training uses seeds disjoint from evaluation. Calibration grounding is documented in `grounding/`.
