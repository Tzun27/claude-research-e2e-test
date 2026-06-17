# Teaching notes — slide-by-slide companion

*Companion to `template.pptx`. Goal: make sure you can speak to every slide confidently. Each section covers what's on the slide, what to say, and any technical concept that might come up in questions.*

---

## Overall framing — what story are we telling?

The talk is structured as a **research-gap proposal**. The arc is:

1. *(Background)* Ride-hailing platforms make three coupled decisions. RL handles them in isolation.
2. *(Related Work)* The two best joint-optimization RL papers each cover only 2 of the 3 decisions. The economics literature (especially Castillo 2025) gives us a welfare yardstick that RL papers don't use.
3. *(The Gap)* Combine those two observations: nobody does all three jointly *and* reports welfare incidence.
4. *(Proposed Approach)* A four-phase plan to attack it.
5. *(Discussion)* Honest caveats (especially the circularity concern), the contribution, and immediate next steps.

The advisor-alignment move: every slide frames the gap in both **CS/RL** terms (what controller, what objective) and **economics** terms (what welfare structure, what strategic-agent assumption). Never one without the other.

---

## Slide 1 — Title

**What's on the slide:** Title, presenter info, and a one-line description of the basis (38-paper lit review).

**What to say:** "I'll be proposing a research gap I've identified after surveying the ride-hailing matching/pricing/rebalancing literature from 2018 to 2026. It's a cross-field gap — RL on one side, economics on the other."

---

## Slide 2 — Outline (full)

Just walk through the five sections quickly. Sets expectations for the structure.

---

## Slide 3 — Outline (Background highlighted)

Transition slide. Just say "Let's start with some background."

---

## Slide 4 — Background: The Three Decisions of a Ride-Hailing Platform

**What's on the slide:** A list of the three operational decisions every ride-hailing platform makes: **matching**, **pricing**, **rebalancing**.

**Concept primer for the audience:**

- **Matching** is the assignment problem at the heart of ride-hailing. At every dispatch tick (often every few seconds), the platform has a set of waiting riders and a set of idle drivers. It computes an assignment — usually a **bipartite matching** — that pairs them. Classical OR uses the Hungarian algorithm or ILP solvers; modern RL learns the *scores* on the edges of that bipartite graph and uses a classical matcher on top.

- **Pricing** sets the base fare and the **surge multiplier**. The surge multiplier is location- and time-varying (Manhattan downtown at 5pm vs. Brooklyn at 2am can differ by 3-4×). Pricing changes demand (higher price → fewer riders) and supply (higher driver pay → more drivers logging on and relocating).

- **Rebalancing** (also called "repositioning") is the *supply-side* counterpart to pricing. Even with good pricing, idle drivers can cluster in the wrong places after dropping off riders. The platform suggests (Uber, Lyft) or commands (in robo-taxi / AMoD systems) where they should drive next.

**What to say:** "These are the three operational decisions. The whole platform-control problem reduces to making these decisions, repeatedly, well." Pause briefly on each.

---

## Slide 5 — Background: Coupled Decisions, But RL Treats Them in Isolation

**What's on the slide:** Two main points: (1) the three decisions are tightly coupled, with three concrete examples of the coupling; (2) RL papers, even the production-deployed ones, almost always handle them in isolation.

**The coupling stories — say them plainly:**

1. Pricing changes demand → demand changes which matchings are feasible. (If you surge to 3×, half the riders cancel, and the matching problem you face next minute is smaller and easier.)
2. Matching outcomes determine where drivers end up idle. If you match a driver to a rider going to JFK, that driver is now in Queens with no riders around — which is a rebalancing problem.
3. Rebalancing changes the supply distribution → that supply change feeds back into the pricing model.

**State of RL in the literature:**

- **Xu 2018** (KDD): Didi's foundational paper. Learn a value function for driver-state, use Hungarian matching on top. Just matching, but the template the whole field copied.
- **Eshkevari 2022** (KDD): Didi international deployment, online state-value iteration with multi-armed-bandit (MAB) graph pruning and safeguards. Just matching.
- **Azagirre 2024** (INFORMS J. Applied Analytics): Lyft's deployed RL matcher. Reports ≥$30M/yr revenue gain (note: switchback experimentation, which has identification caveats — see §6.5 of v5 lit review). Just matching.

So even after seven years of production deployment, the field's deployed systems are still matching-only.

**Concept primer — what is RL in this context?**

Reinforcement learning here means: an agent (the platform, or a driver, or a region) observes a state (current supply / demand / time), takes an action (an assignment, a price, a relocation suggestion), receives a reward (usually some function of revenue or rider/driver welfare), and learns a policy mapping states to actions that maximizes expected cumulative reward over time. **MARL** = multi-agent RL, where each driver or region is its own agent.

---

## Slide 6 — Outline (Related Work highlighted)

Transition. "Now let's see who has gotten closest to the joint problem."

---

## Slide 7 — Related Work: Joint Optimization in RL — The Closest Anyone Has Gotten

**What's on the slide:** The Sun lineage (JDRL @ KDD 2022 → JDRCL @ TKDE 36(7), July 2024) and Zhang 2023, framed as "each version of Sun, and Zhang, each get 2 of 3 decisions — never all three."

### Sun et al. — JDRL (KDD 2022) → JDRCL (TKDE 2024)

This is one research line by the same group (Sun, Jin, Yang, Su, Wang at SJTU & Purdue), with two peer-reviewed versions. Cite both.

**JDRL (KDD 2022) — the conference base:**

- **What it does:** Joint matching + repositioning, MARL, with max-min fairness on driver incomes. Provides a theoretical convergence guarantee for the policy network. Tested on three datasets (Haikou, Chengdu, NYC).

**JDRCL (TKDE 2024) — the journal extension. Three additions over JDRL:**

1. **Budget constraint on repositioning costs.** In real ride-hailing systems, the platform actually pays drivers small monetary incentives to relocate to high-demand areas. JDRCL adds an explicit budget cap on these incentive payments. The optimization becomes "maximize efficiency + fairness *subject to* total relocation incentives ≤ B."

2. **Reframes problem as a Constrained Multi-Objective Markov Game (CMOMG).** Same MARL setup, but now with an explicit constraint term.

3. **Primal-dual iterative training algorithm** with a **sub-linear asymptotic convergence rate** (a stronger theoretical guarantee than KDD 2022's "converges" — JDRCL proves *how fast*).

**Concept primer — what each technical term means:**

- **Max-min fairness:** Instead of maximizing total revenue, you maximize the *minimum* utility across drivers (or driver-groups). Prevents the dispatcher from starving the worst-off drivers. Trade-off: usually costs total efficiency.

- **Markov game:** Multi-agent extension of an MDP. Each agent has its own state-action space; transitions depend on all agents' actions.

- **Multi-objective:** Multiple reward signals (here: efficiency *and* fairness).

- **Constrained:** A separate set of inequalities (here: total budget for repositioning incentives ≤ B) that the policy must satisfy.

- **Primal-dual training:** Standard tool for solving constrained optimization. Augment the loss with Lagrange multipliers (the "dual variables") that penalize constraint violation. Update both the policy (primal) and the multipliers (dual) iteratively. The dual variables effectively learn the right price-per-unit-of-budget.

- **Sub-linear convergence rate:** The training algorithm's error shrinks at a rate slower than 1/√t (or similar). Slower than linear, but still going to zero. Important because non-convex policy networks generically have *no* convergence guarantee at all.

**What to say:**

"Sun's line of work — the conference paper in 2022, the journal extension in 2024 — handles matching and rebalancing jointly under fairness. The 2024 version adds a budget-constraint extension via primal-dual training and proves a convergence *rate*, not just convergence. Either version, the gap is the same: they don't handle pricing. Prices are exogenous inputs to their model."

If the advisor asks "why does the budget constraint matter for your project?" — it's because we'll have analogous *business* constraints in our three-way controller (e.g., total surge revenue can't exceed some cap; total relocation incentives can't exceed a budget). JDRCL's primal-dual framework gives us a template for handling those.

### Zhang 2023 (AAAI)

**One-line summary:** Joint pricing + matching, future-aware ADP, on a NYC ride-pooling simulator.

**Concept primer:**

- **Approximate Dynamic Programming (ADP):** Classical DP solves the Bellman equation exactly via value iteration. That's infeasible when the state space is huge (which it is for ride-hailing — billions of supply/demand configurations). ADP approximates the value function — often with a neural network or a linear basis — and updates it with sampled trajectories. RL is essentially ADP plus some bells and whistles (replay buffers, target networks, exploration policies).

- **"Future-aware":** Zhang's pricing module doesn't just react to current demand; it explicitly accounts for *future* supply-distribution consequences of today's price (the same coupling story from slide 5).

- **Ride-pooling:** Multiple riders share one vehicle along overlapping routes. Adds combinatorial complexity (you're matching not just pairs but small groups of riders to one driver).

**Headline numbers:** +6.4% average revenue, −10.6% fewer vehicles used to serve the same demand, on their NYC simulator.

**What to say:** "Zhang handles pricing and matching jointly. They achieve revenue gains on a simulator. But they explicitly do not handle rebalancing."

**The synthesis bullet at the bottom:** "Both papers stop short of three-way joint optimization. And both report revenue and earnings — neither reports welfare."

---

## Slide 8 — Related Work: The Economics Side — Welfare and Strategic Drivers

**What's on the slide:** Three economics-side papers (Castillo, Besbes, Afeche) plus a clean four-column welfare table with Castillo's published numbers.

### Castillo 2025 (Econometrica 93(5):1811–1854) — *the welfare anchor*

This is the most important paper in the talk. Spend the most time here.

**What Castillo did:** Built a **structural empirical model** of the Houston Uber market and used it to compute the welfare impact of surge pricing.

**Concept primer — "structural empirical model":**

In economics, "structural" means the model is grounded in microeconomic primitives (preferences, technology, behaviors), with parameters estimated from data. As opposed to "reduced form" which just regresses outcomes on treatments without specifying a behavioral mechanism. Castillo's model has:

- Riders with price-elastic demand (estimated from observed price changes)
- Drivers with entry/exit and hours-worked decisions (estimated from observed driver behavior after fare changes)
- A matching technology (estimated from observed wait times)

He estimates the elasticities and then **simulates** the counterfactual: what happens to welfare if Uber used uniform pricing instead of surge?

**The welfare decomposition** (the table on the slide):

- **Total welfare** = rider surplus + driver surplus + platform profit + tax revenue. Surge raises total welfare by **+2.15%** of gross revenue vs. uniform pricing.
- **Rider surplus** = how much riders value the rides minus what they paid. Surge raises this by **+3.57%**.
- **Driver surplus** = earnings minus opportunity cost of driving time. Surge *lowers* this by **−0.98%** (drivers are worse off!).
- **Platform profit** = revenue minus driver pay and other costs. Surge *lowers* this by **−0.50%** (small effect, because the platform redistributes most of the surge revenue to drivers).

**The heterogeneity finding:** Among drivers, those who work long hours are most hurt, and *especially* female drivers — they have different location/speed preferences and less flexibility to relocate to surge zones.

**What to say:** "Castillo's contribution is the first published welfare decomposition that splits the impact of surge across these four parties. It's the yardstick we want our RL controller to be measured against. Notice: Total > 0, Rider > 0, Driver < 0, Platform < 0. That signed pattern is what we'd want to reproduce — it's the welfare *structure*, not the exact numbers, that matters for our project."

### Besbes 2021 (Management Science)

OR-theory paper. Fluid-model abstraction (treats drivers and riders as continuous flows, not discrete agents). Result: when drivers are *strategic* — they relocate to high-surge areas — a platform that prices each location myopically based on local supply/demand is leaving money on the table. **Network-anticipating pricing** (where you consider how supply will move) dominates.

### Afeche 2023 (MSOM)

Extends Besbes with **admission control**: under some regime conditions, the platform should *reject* trip requests it could fulfill. The intuition: serving that ride leaves a driver stranded in a low-demand area, and the platform is better off not serving it.

**The synthesis line on this slide:** "All three economics papers use static or equilibrium models — none of them use learned controllers. So there's a methodological gap on top of the substantive one."

**Concept primer — equilibrium analysis:**

In economics, "equilibrium" usually means a fixed point: agents' best responses given everyone else's strategies form a self-consistent system. Solving for equilibrium analytically is the OR/econ approach. The RL approach is to *learn* policies through trial and error — which can converge to an equilibrium, or not, depending on the algorithm.

---

## Slide 9 — Outline (The Gap highlighted)

Transition. "So with that landscape, what's the actual gap?"

---

## Slide 10 — The Gap: Decisions × Existing Work

**What's on the slide:** A 3×4 matrix. Rows are the three decisions (matching, pricing, rebalancing). Columns are representative papers: Xu 2018, Sun 2024, Zhang 2023, and "Ours". Check marks indicate which decisions each paper handles.

**Reading the matrix aloud:**

- Xu 2018: ✓ on Matching only. The first deployed RL system, sets the bar.
- Sun 2024 (JDRCL): ✓ on Matching, ✓ on Rebalancing. Two of three.
- Zhang 2023: ✓ on Matching, ✓ on Pricing. Two of three.
- Ours: ✓ on all three (planned).

**The point of the matrix:** It makes the gap *visual*. The "Ours" column is the only one that lights up across all three rows. That's the contribution.

**What to say:** "If you draw this matrix for the entire 38-paper corpus, nobody has all three boxes checked. That's the gap."

---

## Slide 11 — The Gap: What No One Has Done

**What's on the slide:** The formal gap statement plus two dimensions of "missing."

**The gap statement, read out loud:**

> "No published work builds a deep-RL controller that handles matching + pricing + rebalancing jointly, AND reports its welfare incidence in the four-way format the economics literature uses."

Two missing dimensions:

1. **Three-way joint optimization:** Even if you unioned Sun 2024's matching+repositioning with Zhang 2023's pricing+matching, that's not the same as a single controller that handles all three coherently — there would be interface problems between the two systems.

2. **Welfare-side evaluation:** Every RL paper reports revenue, completion rate, driver income. Nobody reports the rider/driver/platform/total split. That's not a coincidence — it requires a simulator with strategic agents on both sides, which the RL community doesn't usually build.

**What to say:** "Notice this is a *two-dimensional* gap. If we only did joint three-way RL and reported revenue, that would be one contribution but not the cross-field one. The welfare side is what aligns this with the economics literature and gives it cross-disciplinary weight."

---

## Slide 12 — The Gap: Why This Is the Right Moment

**What's on the slide:** Three pieces of enabling work that have all landed in the last ~2 years, making this project credible *now* in a way it wasn't 24 months ago.

**The three enablers:**

1. **Sun 2024 (TKDE, July 2024):** Gave us the first convergence guarantee for constrained joint matching+repositioning MARL. Before this, even *two-way* joint optimization with fairness was an open question. We'd be extending Sun's framework, not inventing it from scratch.

2. **Castillo 2025 (Econometrica, Sep 2025):** Published the structural elasticity estimates we need to calibrate our simulator. The working paper had been around since 2019 but with different numbers (roughly 2× larger — the published version is the authoritative one to cite). Before this published paper landed, the welfare yardstick was unsettled.

3. **MFG-RL (Mean-Field-Game RL):** Bridge literature between game theory and deep RL. Two key papers:
   - **Guo 2019 (NeurIPS):** Established that mean-field games admit a unique Nash equilibrium under mild conditions, and that a Boltzmann-policy Q-learner converges to it.
   - **Cui-Koeppl 2021 (AISTATS):** Showed that entropy regularization extends MFG-RL to settings where the standard fixed-point iteration would diverge.

   These give us the theoretical scaffolding to handle strategic drivers at city scale.

**Concept primer — Mean-Field-Game RL:**

A Mean-Field Game (MFG) is the limit of an N-player game as N → ∞. Instead of each agent reasoning about every other agent (which is computationally hopeless), each agent reasons about the *average* (mean field) of all other agents' strategies. MFG-RL is the deep-learning version: agents learn policies whose argument is just their own state and the mean field of the population, not the full joint state.

This is the right abstraction for ride-hailing because there are thousands of drivers, and you can't track each one individually — but the *distribution* of drivers across the city is exactly the thing the platform observes and acts on.

**The emphatic line at the bottom (green bold):** "None of them existed together before mid-2024." Translation: this project was simply not buildable a year ago. Now it is.

---

## Slide 13 — Outline (Proposed Approach highlighted)

Transition. "Here's how we'd actually do it."

---

## Slide 14 — Proposed Approach: A Four-Phase Plan

**What's on the slide:** Four phases, each as a dark-green bold header with one or two sub-bullets describing the work.

### Phase 1 — Simulator

Build the testbed. NYC taxi data is the substrate (publicly available; widely used; Sun and Zhang both use it). On top of that we add:

- Price-elastic demand: when surge multiplier goes up, some fraction of riders walk away. The elasticity comes from Castillo's estimates.
- Strategic drivers: drivers decide whether to log on, how many hours to work, and where to relocate. Their decisions respond to expected earnings.
- Heterogeneity: drivers differ on at least one dimension (e.g., full-time vs. part-time; flexibility-of-location) so we can replicate Castillo's heterogeneity finding.

The Phase 1 sanity check: under a *static surge policy* (not RL — just simple price multipliers), does our simulator reproduce welfare numbers close to Castillo's? If yes, the simulator is faithful. If no, we have a calibration bug to fix before doing any RL.

### Phase 2 — Baselines

Re-implement three baselines in the simulator: Xu 2018 (matching only), Sun 2024 (matching + repositioning), Zhang 2023 (pricing + matching). Each baseline runs in the same environment, so the comparison is apples-to-apples.

### Phase 3 — Our Controller

The actual contribution. A three-way joint MARL controller. Key design choices we'd defend:

- **Pricing on a slower timescale than dispatch.** Dispatch happens every few seconds; prices change every few minutes. The two action heads operate at different frequencies. (This avoids the controller "fighting itself" with high-frequency price oscillations.)

- **MFG-RL framing for the driver-side dynamics.** Drivers are strategic but heterogeneous; we represent them as a population whose mean field evolves over time. This is what makes the strategic-agent assumption tractable.

### Phase 4 — Welfare-Structural Comparison

Run every controller in the simulator. For each, compute the four-way welfare decomposition (rider / driver / platform / total). Compare against Castillo's pattern. The success criterion is *structural alignment* — signs and rough proportions — not exact-number match.

Then sweep the heterogeneity parameter. Does our controller still produce a Castillo-like pattern when drivers are very heterogeneous? When they're homogeneous? That sensitivity sweep tells us whether our controller is robust or fragile.

**Concept primer — what "controller" means here:**

In control-theory language, a "controller" is the policy that maps observed state to action. In our case the controller takes in the current city state (supply distribution, demand distribution, time of day, etc.) and outputs three things: a matching, a price vector, and a rebalancing suggestion. Same as a policy in RL.

---

## Slide 15 — Outline (Discussion highlighted)

Transition. "And then the hard questions before we wrap."

---

## Slide 16 — Discussion: Caveats, Contribution, Next Steps

**What's on the slide:** Three bold-header sections.

### Honest scope caveat — the circularity concern (lead with this)

This is the obvious objection. *If our simulator borrows Castillo's elasticities, isn't the welfare comparison rigged? Of course our controller will produce Castillo-like numbers — the environment is built from them.*

It's a real concern. The honest answer:

- We're not testing whether DRL *discovers* Castillo's economics. We're testing whether DRL, *given* Castillo's economics in the environment, can produce a competent welfare-aware controller — which is still a non-trivial claim, because Castillo's static analysis isn't a learned controller and doesn't tell you what to do dispatch-tick by dispatch-tick.

- **Mitigation:** at least one robustness check that varies elasticities away from Castillo's point estimates and asks whether the welfare pattern still holds. If it does, we know the result isn't an artifact of the calibration choice.

**What to say:** "I want to surface this objection myself rather than have it raised in Q&A. It's the most important caveat in the project, and I think we can address it but not eliminate it."

### Contribution if the project succeeds

Three contributions, in order of importance:

1. First three-way joint-RL controller for ride-hailing. (Pure CS/RL contribution.)
2. First DRL result compared to a published structural welfare decomposition. (Cross-disciplinary contribution — this is what makes it interesting to economists.)
3. Methodological bridge between CS/RL and economics/OR communities. (Programmatic contribution — sets up follow-up work.)

### Immediate next steps

What we'd do *this semester*:

1. Finalize simulator design choices (state representation, time discretization, driver heterogeneity parameterization).
2. Reproduce Castillo's welfare numbers under a static surge policy — the Phase 1 sanity check.
3. Re-implement Sun 2024 and Zhang 2023 as baselines.

This sets us up for Phase 3 (the actual controller) starting next semester.

---

## Answering likely questions in Q&A

**Q: Why not just use real Uber Houston data?**
A: It's not public. NYC taxi data is the cleanest publicly-available substrate. We compensate by calibrating to Castillo's structural estimates rather than re-estimating elasticities from NYC.

**Q: How big is the simulator? Computational cost?**
A: NYC taxi data covers ~5 boroughs, ~265 taxi zones. With on the order of 10k–100k drivers and 100k–1M rider trips per day, we're in MARL territory. MFG-RL is what lets us scale — instead of tracking each driver, we track the mean field.

**Q: Why MFG-RL instead of vanilla MARL?**
A: Vanilla MARL doesn't scale to thousands of agents — the joint action space is too big. MFG-RL replaces "every other driver's action" with "the average distribution of drivers," which collapses the dimensionality.

**Q: How does the pricing module differ from Zhang 2023's?**
A: Two differences: (1) it co-exists with a rebalancing module that Zhang doesn't have, so it can shape supply directly rather than only through price; (2) it operates on a slower timescale than dispatch, which Zhang doesn't separate.

**Q: How will you know if it works?**
A: Phase 1 success criterion: simulator reproduces Castillo's welfare numbers under static pricing. Phase 4 success criterion: our controller's four-way welfare split has the same signs and rough proportions as Castillo's, and outperforms each baseline on at least one objective (revenue, fairness, or welfare) without regressing on the others.

**Q: How is this different from JDRCL (Sun 2024)?**
A: JDRCL handles matching + repositioning under max-min fairness; we extend to all three decisions, replace max-min fairness with a welfare-decomposition objective, and add a strategic-driver MFG framing.

**Q: What if the circularity concern is fatal?**
A: Then we reposition the contribution: not "DRL recovers economics" but "DRL operates competently inside an economically-grounded environment." That's still novel and useful. The robustness check is designed precisely to bound how much weight the original framing can carry.

---

## Glossary (quick reference)

- **ADP** — Approximate Dynamic Programming. Numerical approximation of the Bellman value function. RL is ADP plus learning tricks.
- **Bipartite matching** — Pairing problem between two disjoint sets (riders and drivers). Solved by Hungarian algorithm or LP.
- **Equilibrium** — A fixed point of best-response dynamics. In OR/econ, equilibrium is solved for; in RL, it can emerge from learning.
- **Hungarian algorithm** — Classical O(n³) algorithm for optimal bipartite matching.
- **JDRCL** — Sun 2024's name for their joint dispatch + repositioning method with constraints.
- **Markov game** — Multi-agent MDP. Transitions depend on all agents' actions.
- **Max-min fairness** — Maximize the worst-off agent's utility. Standard fairness objective.
- **MARL** — Multi-Agent Reinforcement Learning.
- **MFG / MFG-RL** — Mean-Field Game. Limit of N-agent game as N → ∞, each agent reasons about the population distribution.
- **MFG-Q / MF-AC** — Mean-Field Q-learning / Mean-Field Actor-Critic. Specific MFG-RL algorithms.
- **Primal-dual** — Optimization framework for constrained problems. Iterates on policy (primal) and Lagrange multipliers (dual).
- **Structural empirical model** — Economics-style model where parameters have behavioral meaning and are estimated from data, then used to simulate counterfactuals.
- **Surge multiplier** — Multiplicative pricing factor (1.5×, 2×, etc.) on top of base fare.
- **Switchback experiment** — Alternates treatment across time blocks; has known identification caveats under spillovers (relevant to Lyft's $30M/yr claim).
- **Welfare decomposition** — Splitting total welfare into rider surplus + driver surplus + platform profit + tax revenue.

---

*Good luck with the meeting. If the advisor pushes on anything specific, the most defensible move is to retreat to: "the structural-alignment framing is the contribution; the circularity concern is real and bounded by the robustness check; the cross-field bridge is the lab-aligned angle."*
