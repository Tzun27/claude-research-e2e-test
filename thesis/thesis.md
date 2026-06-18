# Who Benefits from *Learned* Surge Pricing? The Welfare Incidence of Policy-Search Pricing Control under Driver Heterogeneity

*A Master's thesis in Computer Science.*

> **A note on terminology.** The controller in this thesis is a compact policy optimized by **Augmented Random Search (ARS)**, a derivative-free policy-search method, after a per-step deep-RL method (PPO) failed for a diagnosed reason (§4.5). We therefore use "learned control" and "policy-search control" rather than "deep reinforcement learning." The substantive question — whether a *learned* pricing controller reproduces a structural welfare target, and how that depends on its objective — is unchanged by the choice of optimizer.

---

## Abstract

Ride-hailing platforms increasingly set prices, matching, and rebalancing with learned controllers, but these systems are evaluated only on platform-centric metrics — never on *who* gains and loses. Structural economics has answered the latter for a fixed algorithm: Castillo (2025) finds that surge pricing, relative to an optimized uniform price, raises total welfare and rider surplus while reducing driver surplus and platform profit, hurting the most-engaged drivers most. We ask whether a *learned* controller reproduces this welfare-incidence structure, and how it depends on the controller's objective. We build a calibrated spatial ride-hailing simulator with elastic demand, a heterogeneous strategically-responding driver population, and endogenous matching frictions (a wild-goose chase), and an exact four-way welfare decomposition; controllers are optimized by Augmented Random Search, a derivative-free policy-search method we adopt after a per-step deep-RL method (PPO) failed for a diagnosed credit-assignment reason. Decomposing surge's effect, we find that the **price-variation channel** — spatio-temporal price variation at a fixed average — robustly improves welfare (+2.3% of gross revenue) and helps riders most (+1.9%) while helping drivers and platform only slightly (≈+0.2% each), exactly the structure Castillo attributes to allocative efficiency. The **price-level channel**, and hence the full incidence, is governed not by surge but by the controller's *objective*: [*FILL: the cross-objective contrast — profit vs welfare-weighted vs throughput*]. The most-engaged (long-hours) drivers bear [*FILL*] of the driver-side incidence, via a timing/exposure mechanism; a fairness penalty traces an efficiency–equity frontier on which the driver loss can be neutralized at a welfare cost of [*FILL*]. The central message is that the distributional incidence of learned surge is a property of the objective, not of surge — making the choice of training objective an implicit, and currently unmeasured, distributional policy.

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

an identity the simulator verifies to machine precision (Section 5.0). Following Castillo we report welfare components as percentages of **gross revenue** $\sum_{\text{served}} p$.

### 3.3 The four platform objectives

The controller is trained to maximize one of four per-episode objectives, which constitute the RQ2 treatment:
- **Profit:** $\sum \gamma p$ (platform commission).
- **Throughput:** number of completed trips (a proxy for the order-response / completion-rate objectives that deployed systems optimize).
- **Welfare:** total welfare $W$ (a social planner).
- **Welfare-weighted:** $\alpha_\pi\,\text{PP} + \alpha_R\,\text{RS} + \alpha_D\,\text{DS}$, the analog of Castillo's platform, which maximizes profit plus weighted rider and driver surplus. We use $\alpha_\pi=0.4,\ \alpha_R=1,\ \alpha_D=0.3$ — heavily rider-weighted, with $\alpha_\pi+\alpha_D$ at the upper end of Castillo's identified range ($<0.7$).

### 3.4 The uniform-pricing counterfactual

To isolate the welfare incidence of **price flexibility** — Castillo's exact comparison — we fix the matching and rebalancing levers to a common policy and compare two pricing regimes: a **uniform** policy that applies a single constant multiplier to the whole market, versus a **surge** policy that sets multipliers per zone and per epoch (the learned, policy-search controller of §4.5). We decompose the surge-vs-uniform effect on each welfare component $C\in\{W,\text{RS},\text{DS},\text{PP}\}$ into two channels. Let $\bar m$ be the surge policy's trip-weighted average multiplier, $m^\*$ the platform-objective-optimal uniform multiplier, and $C(\text{surge})$, $C(\bar m)$, $C(m^\*)$ the component under surge, under a flat price at the surge's own average, and under the optimal uniform price. Then

$$\underbrace{C(\text{surge}) - C(m^\*)}_{\text{total effect (vs optimal uniform)}} \;=\; \underbrace{\big[\,C(\text{surge}) - C(\bar m)\,\big]}_{\text{price-variation channel (RQ1)}} \;+\; \underbrace{\big[\,C(\bar m) - C(m^\*)\,\big]}_{\text{price-level channel (objective-dependent, RQ2)}}.$$

The **variation channel** is a mean-preserving spread (surge vs. a flat price at the *same* average) and isolates allocative efficiency plus matching gains; it is calibration-robust. The **level channel** is the welfare effect of moving the average price from the platform's optimum $m^\*$ to $\bar m$, and its sign is set by the objective (via $m^\*$). All components are reported as percentages of gross revenue.

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

Parameters are grounded in the corpus (grid size from Gammelli 2021; time/fleet/episode structure from Lin 2018 and Zhang 2023; fare structure and commission from Castillo 2025). The two free demand parameters (the WTP ratio's location and scale) are set so the short-run demand price-elasticity — measured in a flooded-supply regime, isolating the pure price response — is $\approx 0.25$, close to Castillo's estimated $0.268$, with flooded acceptance $\approx 0.77$. The fleet size (400) places the market in a supply-constrained regime (match rate $\approx 0.58$ at multiplier 1) where surge has room to reallocate scarce capacity. The congestion-delay coefficient ($0.8$) gives the matching technology a genuine wild-goose chase, so low prices are costly (long waits, drivers tied up) — the qualitative mechanism Castillo identifies.

We deliberately do *not* tune any parameter to force the welfare-weighted platform's optimal uniform price to match Castillo's level; doing so would rig the central comparison. The resulting grid-searched optimal uniform multipliers form a clean spread that differs by objective — profit $\approx 3.4$, welfare $\approx 2.0$, throughput $\approx 1.2$, welfare-weighted $\approx 0.9$. **One divergence from Castillo must be stated plainly here:** his welfare-weighted platform optimum is $1.174$ (above the base fare), whereas in our proxy it sits near the base fare ($\approx 0.9$). Because Castillo's driver- and platform-loss arises from surge pricing *below* a high optimal uniform price, this divergence means the price-*level* channel is weak for the welfare-weighted objective in our market and strong for the higher-optimum objectives (welfare, profit). This is exactly why we test RQ1 through the **decomposition** of §3.4: the price-*variation* channel (mean-preserving spread) is calibration-robust, while the price-*level* channel is read off per objective. The four-way welfare accounting satisfies the consistency identity $W = \text{RS}+\text{DS}+\text{PP}$ to a maximum relative error of $\sim10^{-16}$ across pricing regimes and seeds (including with the rebalancing transfer active).

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

**Demand slopes down and is inelastic in the right range.** Requests fall monotonically with price ($7{,}717\to6{,}783\to5{,}302$ as the multiplier rises $1.0\to1.5\to2.5$). The short-run demand price-elasticity, measured in a flooded-supply regime so it reflects the pure price response (Castillo's demand-elasticity definition), is $0.251$ — close to his estimated $0.268$ — with a baseline acceptance of $0.766$. At the supply-constrained operating point (fleet 400, multiplier 1) the match rate is $0.578$, acceptance is $0.692$, and the mean rider wait is $0.89$ epochs.

**The wild-goose chase is a real force.** The supply–demand imbalance concentrates almost entirely at demand peaks: the correlation between per-epoch abandonment and demand is $0.962$, with $\sim\!177$ abandonments per peak epoch versus $\approx\!0$ per trough epoch. A dispatch-radius sweep confirms the intensive-margin chase: widening the radius lengthens mean pickups and, beyond radius $\approx2$, *lowers* welfare because drivers are tied up serving distant riders. The congestion-delay term makes low prices costly, sustaining an interior platform-optimal price (§4.4).

**Objectives genuinely differ, and the controller is a fair instrument.** The grid-searched optimal uniform multiplier rises with the platform's emphasis on its own revenue — welfare-weighted $\approx0.9$, welfare $\approx2.0$, profit $\approx3.4$ — so the four objectives genuinely price differently. As a capacity check, the ARS controller recovers each objective's optimum: for the profit objective it reaches the profit-maximizing price level (learned mean surge $\approx3.0$, objective within $\sim0.1\%$ of the grid-searched uniform optimum), and for the welfare-weighted objective it slightly *exceeds* the optimal uniform via spatial variation — confirming the compact policy is expressive enough to find the objective-specific optimum rather than being an artificially weak baseline.

### 5.1 RQ1 — Does learned surge reproduce the Castillo incidence structure?

We test the two channels of §3.4 (Table `rq1_mean_preserving_spread.md`, Figure `welfare_incidence_price.png`). The **price-variation channel** — the welfare-weighted controller's learned surge versus a flat price at the *same* average multiplier — yields, as a percentage of gross revenue: total welfare [W1], rider surplus [R1], driver surplus [D1], platform profit [P1]. [INTERP: state whether this matches Castillo's "surge dominates at every average, helping riders most while helping drivers/platform slightly" — the allocative-efficiency structure.] This channel is calibration-robust and does not depend on the platform's objective.

The **price-level channel** — moving the average price from the platform's optimum to the surge average — is where Castillo's driver/platform losses live. [INTERP: report, for the welfare-weighted objective, the surge average versus the optimal uniform $m^\*\approx0.9$, and what this implies for the level channel's sign; relate to the proportions Castillo finds (rider gain $\approx 3.6\times$ the driver loss).] The headline test of *proportions* — does the rider gain exceed the driver and platform losses, as in Castillo? — is [INTERP].

### 5.2 RQ2 — Objective-dependence of the welfare incidence (central result)

This is the load-bearing result. Table `rq2_incidence_by_objective.md` reports the welfare *shares* each objective's learned surge controller delivers (as % of its own gross revenue):

[TABLE: rq2 — per objective: surge avg multiplier; rider %GR; driver %GR; platform %GR; total %GR.]

The same surge *mechanism*, optimized for different objectives, produces opposite incidence. A **profit** controller prices near the willingness-to-pay ceiling (mean multiplier $\approx 3.4$) and compresses rider surplus to [PR]% of gross revenue, capturing the rest as driver and platform surplus — it *distorts* the Castillo structure toward extraction. A **welfare-weighted** controller (Castillo's analog) prices near base (mean multiplier $\approx$ [WWm]) and delivers rider surplus of [WWR]% — reproducing the rider-favoring incidence. **Throughput** and **welfare** sit between [INTERP]. The cross-objective spread in rider share — from [PR]% (profit) to [WWR]% (welfare-weighted) — is the quantitative statement that *the incidence is a property of the objective, not of surge.* A platform optimizing a throughput/GMV metric, as deployed systems do, lands at [INTERP] — a distributional outcome its own metric cannot reveal.

### 5.3 RQ3 — Heterogeneous incidence across driver types

Table `driver_incidence.md` decomposes the driver-side effect by type (per-capita surplus change, surge − uniform). [INTERP: state whether full-time (low-reservation-wage, long-hours) drivers experience the largest per-capita loss/gain change relative to part-time and casual drivers, and by how much.] The mechanism is timing/exposure: full-time drivers are online across low-surge off-peak periods, so their average earnings track the whole day, while casual drivers concentrate at peaks; [INTERP: relate to whether surge therefore shifts surplus toward peak-concentrated drivers and away from the always-on full-timers, reproducing Castillo's "long-hours drivers hurt most" via the same when-they-work channel rather than an elasticity channel]. A policy-inspection diagnostic (Figure `policy_price_welfare_weighted.png`) confirms the controller learns a genuine peak/off-peak surge spread [INTERP], without which this channel would be inert.

### 5.4 RQ4 — The efficiency–equity frontier

Imposing a driver-earnings fairness penalty (max-min across types) and sweeping its weight traces an efficiency–equity frontier (Figure `fairness_frontier.png`). [INTERP: as the weight rises, the across-type earnings dispersion falls from [G0] to [G1] while total welfare falls from [W0] to [W1]; the driver-side loss can be neutralized at a welfare cost of [COST]% of gross revenue.] This operationalizes the corpus's "fairness as a constraint" consensus and quantifies its price in this market.

### 5.5 Three-way control (Gap A0) and scope conditions

**Three-way control.** Allowing the controller to set matching (dispatch radius) and rebalancing in addition to price — against a uniform baseline that also optimizes the radius (so the comparison still isolates price flexibility) — gives the incidence in Table `threeway_vs_price.md`. [INTERP: state whether adding the two levers preserves the price-only incidence structure or shifts it, for the welfare-weighted and profit objectives; note that with the costed rebalancing transfer the welfare accounting still closes.]

**Scope conditions.** The sensitivity sweep (Table `sensitivity.md`) re-runs the welfare-weighted price experiment under a tighter fleet (300 drivers) and a more dispersed willingness-to-pay (more elastic demand). [INTERP: state across which regimes the qualitative incidence (sign pattern of the decomposition) is preserved and where it weakens, identifying the boundary of the regime in which the conclusions hold.] Together with the objective sweep, this delimits *when* the Castillo-type structure emerges in a learned market: [INTERP one-sentence boundary statement].

---

## Chapter 6 — Discussion

### 6.1 Interpretation: incidence is a property of the objective

The organizing finding is that *who benefits from surge is determined by what the controller is told to maximize, not by surge pricing itself.* The same learned surge *mechanism* — raise the multiplier where and when demand outruns supply — produces a rider-favoring redistribution under a welfare- or rider-weighted objective and a platform-favoring extraction under a profit objective. This is best seen through a decomposition of surge's effect into two channels: a **price-variation effect** (surge versus a flat price at the same average), which improves total welfare and helps riders most through allocative efficiency, and is essentially objective-independent; and a **price-level effect** (the gap between the platform-optimal flat price and the surge average), whose sign and size are set entirely by the objective. Castillo's headline incidence — riders up, drivers and platform down — is the sum of a small positive variation effect and a downward level shift that arises *because his platform prices above the surge average*. A profit-maximizing controller prices at the willingness-to-pay ceiling, so its level effect runs the other way.

This has a direct implication for practice. Deployed ride-hailing RL is evaluated on platform-centric metrics — gross merchandise value, completion rate, driver income. Our results show that a controller maximizing such a metric can post excellent numbers on it while moving welfare between riders, drivers, and the platform in a direction the metric cannot see. The distributional consequence of the objective is real and large — riders capture on the order of [PR_W]% of total welfare under a profit-maximizing controller versus [TR_W]% under a throughput-maximizing one, on the same market with the same surge mechanism — and is currently unmeasured in the deployment literature. Choosing the objective is therefore an implicit distributional policy choice, and our four-way decomposition makes it explicit.

### 6.2 Alignment with and divergence from Castillo

[*FILL after results: where the learned controller's incidence aligns with Castillo's signs/proportions and where it diverges.*] Two points hold independent of the exact numbers. First, the *mechanism* by which the variation effect helps riders — surge screens scarce capacity toward higher-value riders and curbs the peak crunch — is the same allocative-efficiency channel Castillo identifies, and drivers do not share in it for the same reason (meter-based pay). Second, our market generates the supply–demand imbalance primarily on the extensive margin (peak abandonment) plus a reduced-form congestion delay, whereas Castillo's stated main channel is intensive-margin pickup time; the alignment we claim is at the level of welfare *incidence*, not the micro-mechanics of matching (see §6.3).

### 6.3 Threats to validity

We are deliberate about what this study does and does not establish.

**Sim-to-real.** All results are on a calibrated proxy, not a real platform. The simulator abstracts a city to a $5\times5$ grid, demand to a bimodal commute, and the driver population to three behavioral types. The *direction* and *mechanism* of the findings are the contribution; the *magnitudes* should not be read as predictions for any real market. We never claim to replicate Castillo's Houston numbers — the underlying data are proprietary — only to test whether learned control reproduces the *structure* he documents.

**The designed-in signs.** For the welfare-weighted (Castillo-analog) objective, the broad sign pattern (rider-favoring) is partly a consequence of the objective's weights, which place weight 1.0 on rider surplus. This is faithful to Castillo, who likewise backs out the platform's weights, but it means RQ1's value is not in the signs per se. The genuinely informative results are (i) the *cross-objective contrast* (RQ2) — the same surge mechanism produces opposite incidence under a profit objective — and (ii) the *decomposition* of the effect into a calibration-robust price-variation channel and an objective-dependent price-level channel.

**Single calibration and the channel abstraction.** One free parameter (the congestion-delay coefficient) is set so the matching technology has a genuine wild-goose chase that makes low prices costly — the qualitative mechanism Castillo invokes — without tuning the welfare-weighted platform optimum to his level (it lands near the base fare, $\approx0.9$, a divergence we own in §4.4 rather than engineer away). The sensitivity sweep (§5.5) tests whether the qualitative incidence survives across fleet sizes and demand elasticities; the conclusions should be read as holding *within* the regime where those qualitative features hold. A substantive divergence from Castillo is worth naming plainly: his stated "main driver" is an *intensive-margin* pickup-time chase, whereas in our calibration the supply–demand imbalance manifests largely on the *extensive margin* (peak abandonment) plus a reduced-form congestion delay. Our welfare gains therefore flow partly through a different physical channel than his; the structural *alignment* we test is at the level of incidence signs and proportions, not the micro-mechanics of matching.

**The optimizer and policy class.** The controller is a compact, location-invariant policy optimized by ARS, not a deep network. This is a deliberate, justified choice (PPO failed for a diagnosed reason, §4.5), but it is a weaker function class, so a *failure* to reproduce a structure could in principle reflect controller capacity rather than the market. We mitigate this by verifying that the controller recovers each objective's grid-searched uniform optimum (so it is not an artificially weak baseline) and report this as a capacity check. ARS finds a good policy, not a certified global optimum; reported policies are the best over the training seeds.

**Driver model.** Drivers follow a fixed behavioral rule (earnings-following relocation, reservation-wage participation) rather than co-learning. This matches Castillo's "platform optimizes, drivers respond in equilibrium" structure and keeps the environment stationary, but it omits driver-side strategic learning and any rider-side learning. The heterogeneity that drives RQ3 is a *timing/exposure* construction; we attribute the by-type incidence to it but do not claim it is the only possible mechanism.

**Statistics.** Core cells use three training seeds and sixteen evaluation seeds; confidence intervals are over evaluation seeds. Three training seeds is modest; where we claim an objective "distorts" versus "reproduces," we report the per-train-seed objective values so the reader can see the spread.

---

## Chapter 7 — Conclusion and future work

This thesis took the question "who benefits from surge pricing?" — answered for a fixed proprietary algorithm by structural econometrics — and asked it of *learned* controllers, the kind increasingly deployed in practice. The contribution is not a new control algorithm but a new *evaluation*: measuring the four-way welfare incidence of a learned pricing controller against a published structural target, on a calibrated proxy market with a heterogeneous, strategically-responding driver population, and decomposing surge's effect into a calibration-robust price-variation channel and an objective-dependent price-level channel.

### 7.1 Summary of findings

> *[FILL after results: one paragraph stating the RQ1–RQ4 answers with the headline numbers. Headline: the welfare incidence of learned surge is a property of the controller's objective, not of surge per se; the variation channel helps riders most while the level channel's sign is set by the objective; the most-engaged drivers bear the largest losses; and a fairness penalty buys driver equity at a stated welfare cost. This reframes the deployed-systems practice of optimizing platform-centric metrics as an implicit, unmeasured distributional choice.]*

### 7.2 Future work

Several extensions would sharpen or broaden the result. **Intensive-margin frictions.** Our market generates imbalance mainly through peak abandonment and a reduced-form congestion delay; an explicit pickup-time/queueing model (drivers tied up on long pickups, riders queuing) would let the proxy match Castillo's stated main channel and test whether the incidence is channel-invariant. **Decoupled price ceilings.** The surge cap and the willingness-to-pay cap coincide in our calibration; sweeping them independently would confirm the profit-objective extraction is not an artifact of that coincidence. **Co-learning drivers.** Drivers here follow a fixed behavioral rule; making them learning agents (a Stackelberg or mean-field-game equilibrium between platform and drivers) would test robustness of the incidence to strategic driver adaptation. **Multi-platform competition.** Castillo's clean single-platform setting is matched here; competition would change both the optimal price and the incidence. **Real-data transfer.** The decomposition evaluation could be ported to a public-data simulator (e.g., NYC taxi) with a calibrated demand-elasticity and driver-heterogeneity model, moving from structural *alignment* toward structural *estimation*. **Objective design.** Since incidence is a property of the objective, an explicit fairness- or welfare-shaped objective — possibly an LLM-as-reward-designer over the four-way decomposition — is a natural next controller, and a stronger function class (a deep policy under a sample-efficient optimizer) would test whether richer controllers sharpen the incidence. Finally, more training seeds and a tuned competitive surge baseline would tighten the statistical claims.

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
