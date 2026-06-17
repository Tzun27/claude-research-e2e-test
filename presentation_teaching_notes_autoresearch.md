# Teaching notes — `rideshare_auto.pptx` (autoresearch reframe)

Companion to the 13-slide deck. One section per slide. Every section follows the same structure:

1. **What's on the slide** — the 5-line summary in plain English
2. **The story you're telling** — the actual message, not jargon
3. **Terms you might get asked about** — plain definitions
4. **Likely questions** — what to expect and how to answer
5. **Tone / framing tips** — where to be confident, where to hedge

Goal: you should be able to read each section once and feel ready to teach it.

---

## Slide 1 — Title

**On the slide:** *Agentic Autoresearch for Joint Decision-Making in Ride-Hailing Platforms* + "A research-direction proposal" + grounding line + "Group meeting".

**The story:** Two beats.
1. "I'm proposing a research direction, not a finished result."
2. "It sits at the intersection of two things — the 38-paper ride-hailing review we built, and the agentic-autoresearch framework Senior X used in their hydrology thesis."

**Why this framing matters:** You're anchoring on two things — a literature review you did, and a methodology your lab has already validated. That's evidence on two axes; it signals you're not pivoting on a whim.

**Term to know:**
- **Agentic autoresearch** = an AI system made of multiple specialized "agents" (LLM-driven programs each with a role) that *automatically* does parts of the research process — picking methods, designing experiments, evaluating results. The "autoresearch" part means the system itself decides what to try, not just a researcher with a tool.

---

## Slide 2 — Outline

Five sections. Pause for half a beat before moving on.

---

## Slide 3 — Background: The Three Coupled Decisions

**On the slide:** Every ride-hailing platform makes three decisions, continuously, on the map of a city:
- **Matching** — who picks up whom
- **Pricing** — what to charge (base fare + surge multiplier)
- **Rebalancing** — where idle drivers should go next

**The story:** These three decisions can't be treated as independent — they feed into each other. Pricing changes how much demand there is. Matching changes where drivers end up idle. Rebalancing changes where the supply is for the next round of pricing decisions. It's a loop.

**Terms to know:**
- **Surge multiplier** = a number greater than 1 applied to the base fare when demand outstrips supply. A "2.0× surge" means the trip costs twice as much as usual.
- **Idle driver** = a driver in the app, available to take rides, but not currently on a trip.
- **Rebalancing / repositioning** = same thing — telling (or nudging) idle drivers to drive themselves to areas where rides are more likely.

**Likely question — "What about driver onboarding, fleet sizing, route planning?"**

→ Those are *strategic* decisions on longer timescales (days, weeks, months). Out of scope here — we focus on the *operational* loop that runs minute-to-minute.

---

## Slide 4 — Background: Two Implicit Choices, Made Without Context

**On the slide:** Every ride-hailing paper makes two decisions implicitly:
1. **What to optimize** — what counts as success (platform revenue, matching rate, fairness, welfare decomposition, …)
2. **How to optimize it** — which method (deep RL, integer programming, mean-field games, contextual bandits, fluid models, …)

The 38-paper corpus shows striking variety in both. Both choices usually come from what the researcher is already trained in, not from carefully reading the specific problem.

**Working hypothesis:** If we make both choices explicit and tie them to the city's context, the result is easier to justify and easier to reuse elsewhere.

**The story:** "Look at the literature. The same problem — make matching/pricing/rebalancing decisions in ride-hailing — has been tackled with at least five totally different math/CS approaches. And researchers have optimized at least four different things. Nobody systematically argues which choice is right for which situation; they just pick what they're trained in. That's the gap we want to attack."

**Terms to know (the five method families — useful here AND on slide 8):**
- **Deep RL (deep reinforcement learning)** = a computer program that learns by trial and error which actions work best in which situations. It uses a neural network to do the learning. Strength: handles huge state spaces. Weakness: needs lots of data, hard to interpret.
- **Integer Programming (ILP — integer linear programming)** = a math-optimization technique for problems where you have to make yes/no or pick-one-of-N choices subject to hard constraints. "Find the assignment of drivers to riders that minimizes total wait time, given that each driver can only take one rider." Strength: provably optimal answers, hard constraints respected. Weakness: doesn't learn from data; doesn't look ahead in time.
- **Mean-Field Games (MFG)** = a math framework for problems with so many agents (e.g., 50,000 drivers in a city) that you can treat them as a *continuum* rather than individuals. Each agent reacts to the *average behavior* of the crowd. Strength: scales to huge populations. Weakness: assumes everyone is interchangeable.
- **Contextual Bandit** = a slimmed-down version of RL. At each step you pick one of several options (e.g., a price multiplier), see what happens, learn. Doesn't worry about long-term consequences — focuses on the immediate next reward. Strength: fast to train, easy to deploy. Weakness: no long-horizon planning.
- **Fluid Model** = treat the flows of riders and drivers as continuous fluids (like water in pipes) rather than discrete trips. Useful for analytical reasoning at the city level. Strength: clean equilibrium analysis. Weakness: hides what happens to individual drivers/riders.

(Each of these is appropriate for *different* situations. The point of the proposal is that nobody currently picks among them systematically.)

**The four common "what to optimize" choices:**
- **Platform revenue / GMV (gross merchandise volume)** = total dollars flowing through the platform. What businesses tend to maximize.
- **Matching rate / success rate (SR)** = fraction of ride requests that successfully get matched to a driver. Operational metric.
- **Fairness** = something like "make sure no driver is consistently the worst-off." Various formal definitions.
- **Welfare decomposition** = break the total benefit into pieces (rider benefit, driver benefit, platform benefit) and look at each piece separately. The economist's preferred answer.

**Likely question — "Isn't this just AutoML?"**

→ No. AutoML picks the *method* within a fixed problem class (e.g., "find the best image classifier"). We're also picking what the *objective* is in the first place — Phase 2's expert agents do that part. That has no AutoML analogue.

**Likely question — "Aren't researchers already making these choices based on data?"**

→ Yes, but implicitly — in their heads, based on their training. The proposal is that making the choices *explicit* gives you four things you don't otherwise get: reproducibility, transferability across cities, an audit trail of *why* this method was chosen, and the ability to swap one expert agent for a better one without redoing the rest.

**Honest framing — why this hypothesis instead of "different cities need different methods":** The cross-city-transfer claim is suggestive but not actually established in the literature. Stating it as fact hands the advisor a cheap challenge ("are you sure?"). The "two implicit choices" framing is directly supported by your own 38-paper review — you don't need to speculate.

---

## Slide 5 — Related Work: RL in Ride-Hailing — Including the Joint-3 Baseline

**On the slide (five bullets):**
- **Production-deployed RL (matching only):** Xu 2018 (Didi) → Tang 2019 → Eshkevari 2022 → Azagirre 2024 (Lyft). Seven years of single-platform success reports.
- **Joint-2 RL:** Sun JDRL (KDD 2022) → JDRCL (TKDE 2024) — joint matching + repositioning with fairness constraints. Zhang 2023 — joint pricing + matching on an NYC simulator.
- **Joint-3 RL — JPDR (TKDE 2024):** pricing via contextual bandit + dispatching/repositioning via SAC-based multi-agent RL, trained jointly. Outperforms strong baselines on platform revenue and success rate.
- **What JPDR doesn't do:** only tested on one city (NYC 2016 green taxi), only reports platform-side scalar metrics, doesn't model that different drivers respond differently, doesn't break welfare down by stakeholder.
- **Note (italic):** the 38-paper review also contains non-RL approaches (ILP, MFG-RL, fluid models, structural economics, LLMs). We zoom in on RL here because JPDR is the closest competitor; the variety of approaches is what motivates this proposal.
- **Implication:** joint-3 is no longer an open feasibility question. What's still open is *method choice* and *welfare incidence*.

**The story:** "Here's the RL family in our corpus. Seven years ago people deployed RL just for matching. Then pairs of decisions got joined together. Last year (TKDE December 2024), JPDR did all three jointly. So that question — *can we do this?* — is closed. But JPDR has clear holes: only NYC, only platform metrics, no driver differences, no welfare breakdown. Those holes are what's still open."

**Terms to know:**
- **Joint-2 / Joint-3** = how many of the three decisions (matching, pricing, rebalancing) a paper handles simultaneously. Joint-2 = two of the three. Joint-3 = all three.
- **JPDR** = the paper's acronym: Joint optimization of Pricing, Dispatching, and Repositioning. ("Dispatching" is just "matching" in their vocabulary.)
- **JDRL / JDRCL** = Sun's acronyms — Joint Dispatching and Repositioning Learning (KDD 2022 version) and the Constrained extension (TKDE 2024 version). The Constrained version handles fairness as a *hard constraint* rather than a soft preference.
- **SAC (Soft Actor-Critic)** = the most popular modern RL algorithm for problems where the action you take is a *smooth dial* (like "set surge to 1.73×") rather than a *small set of buttons* (like "pick one of {1×, 1.5×, 2×}"). Under the hood it uses two neural networks working together: an **Actor** that decides what to do, and a **Critic** that scores how good each situation is. The Actor learns from the Critic's scores. The **"Soft"** part means the algorithm deliberately keeps a little randomness in the Actor's choices instead of always picking the single best-looking option — that randomness keeps it exploring better options instead of locking onto the first decent one. *Why JPDR uses it:* dispatch and repositioning have continuous action spaces (where exactly should this driver go?), and SAC handles those stably without much hyperparameter tuning.
- **Multi-agent RL (MARL)** = RL where multiple agents learn simultaneously and interact with each other. In ride-hailing, "agent" usually means either *each driver* or *each grid cell on the city map*.
- **Contextual bandit** = the slimmed-down RL flavor (defined above). JPDR uses one for pricing because pricing decisions don't need long-horizon planning the way dispatching does.
- **GMV** = Gross Merchandise Volume — total dollars of completed trips. Platform-side metric.
- **SR** = Success Rate — fraction of ride requests that get matched to a driver. Platform-side metric.
- **Welfare decomposition by stakeholder** = breaking the total well-being change into rider, driver, and platform pieces separately (the Castillo table on the next slide). JPDR doesn't do this.
- **Driver heterogeneity** = the fact that different drivers respond differently to the same dispatch / price signal (because of hours, demographics, location preferences). JPDR treats every driver as identical and obedient.
- **Primal-dual training** (used by Sun JDRCL) = a technique for training under a hard constraint (e.g., "fairness above threshold X") by simultaneously learning the policy AND learning how much weight to put on the constraint. Lets you tighten or loosen the constraint without retraining from scratch.

**Likely question — "So isn't your proposal just incremental over JPDR?"**

→ No. JPDR builds *a controller*. We're proposing to build the *agent that builds the controller*. Different problem, different output, different evaluation.

**Honest framing:** Don't bash JPDR — it's a strong paper. The differentiation is real but it's about *complementary* questions, not "JPDR is wrong." Stay generous; the contribution argument is sharper that way.

---

## Slide 6 — Related Work: The Economics Anchor — Castillo (2025)

**On the slide:**
- Castillo (2025), *Econometrica* 93(5):1811–1854 — a structural economic model fitted to Uber's Houston data.
- Decomposes the welfare effect of surge pricing across four parties (the table).
- Female drivers and long-hours drivers bear most of the driver-side loss — not spread evenly.
- It's a published, peer-reviewed *target* — Phase 4 of our pipeline measures against it, and Phase 2's Economics agent is grounded in it.
- No DRL ride-hailing paper in the corpus produces anything comparable.

**Plain-language reading of the table:**

Compare two versions of Uber in Houston — one with surge pricing on, one with flat prices. Castillo's model says:

- **Total: +2.15%.** The whole system runs slightly more efficiently with surge. Riders wait less, drivers idle less, more trips happen at the right time and place. A small but real efficiency gain.
- **Riders: +3.57%.** Riders come out ahead. They pay *more* per trip, but they get matched *faster* — and the time saved is worth more to them than the higher fare.
- **Drivers: −0.98%.** This is the surprise. Per-trip earnings go up under surge — but more drivers chase the same peak, each one ends up with fewer high-fare trips, and idle time eats into hourly take-home. Net hourly income drops slightly.
- **Platform: −0.50%.** And this is the bigger surprise. Even Uber itself loses a little. Surge isn't a money pump for the platform; total ride volume drops enough that their cut shrinks.

**The story Castillo is telling — say this out loud:**

The popular story is "Uber uses surge to gouge riders and enrich the platform (or its drivers)." Castillo's data shows the opposite is closer to the truth. Surge creates a small overall efficiency gain — and that gain goes to *riders*, paid for by *drivers* (and slightly by Uber itself). It's a quiet redistribution from supply side to demand side, dressed up as a market-clearing mechanism.

**The kicker — links forward to slide 8:**

That −0.98% driver loss is *not* spread evenly. Drivers who *can* chase surge (flexible schedule, can move into hot areas, can do long shifts at peak times) actually come out ahead. Drivers who *can't* flex — long-hours drivers already maxed out on hours, and female drivers whose driving patterns don't peak-chase as well — bear the entire loss and then some. So the headline isn't just "drivers lose a little"; it's that a small overall gain is being financed by a specific, identifiable subset of drivers.

**Terms to know:**
- **Structural model** = a model built from economic *theory* (people respond to prices, drivers respond to wages, supply meets demand at equilibrium). Contrast with a "reduced-form" model that just fits patterns in the data without theoretical structure. Structural models predict what would happen in *counterfactual* worlds (like "what if surge were turned off?"), which is why Castillo can tell you the numbers in the table.
- **Welfare** = total well-being in dollar-equivalent units. "Total welfare = +2.15%" means the whole system is 2.15% better off (in dollar terms, expressed as a fraction of total revenue) under surge than under flat prices.
- **Rider surplus / driver surplus** = economics jargon for "net benefit, in dollars." For riders, it's "what the ride was worth to me minus what I paid." For drivers, it's "what I earned minus the opportunity cost of my time." On the slide we'll likely just write "Riders' gain / Drivers' gain" — same numbers, plainer English.
- **Platform profit** = standard accounting — revenue minus costs.
- **Counterfactual** = the alternative scenario being compared against. Castillo's "uniform pricing counterfactual" = "what would happen if Uber turned surge off and charged flat fares."

**Why this econ paper is in a CS proposal — answer in one sentence:**

Castillo gives us a *falsifiable target* — a published number for what a welfare-aware controller should reproduce. Almost no DRL paper has anything like this; they optimize platform revenue or matching rate, which can look like an improvement while quietly recreating exactly the incidence pattern Castillo documented. Decomposing welfare the way Castillo does is how you tell the difference.

**Likely question — "But that's Houston, not NYC/Taipei/etc."**

→ Correct, and we don't pretend otherwise. The target we'd measure against is the *structure* of the decomposition — the signs and rough proportions — not the literal numbers. The whole point of Phase 1's City Expert is to handle cross-city differences.

**A note on the labels:** "Rider/driver surplus" is welfare-economics jargon. On the slide I'd recommend using "Riders' gain / Drivers' gain / Platform's gain" with the same numbers — parallel English is clearer for a CS audience than the technical terms.

---

## Slide 7 — Related Work: Agentic Autoresearch — A Methodological Precedent

**On the slide:** A four-phase agentic pipeline, originally from a recent lab thesis on hydrologic hazard prediction.
- **Phase 1 (one-shot):** a domain expert agent ingests human-defined input and produces a region context.
- **Phase 2 (iterative):** specialized expert agents propose how to define the objective; a Coordinator reviews and scores; the loop runs until they converge.
- **Phase 3 (per sub-task):** autoresearch agents discover the best optimization method per sub-task.
- **Phase 4:** an Aggregator compiles all the sub-task outputs into the final result.
- Advisor characterization: *"the most interesting research direction I've heard this year."*

**The story:** "Senior X built this for hydrology. The lab — and specifically the advisor — has said this is exciting work. We're proposing to take the *same methodology* and apply it to ride-hailing, because ride-hailing has exactly the right ingredients for this kind of agentic system: real method diversity, a good evaluation target (Castillo), and clear context-dependence (cities differ)."

**Terms to know:**
- **Agentic pipeline** = a chain of specialized AI agents (each an LLM with a specific role and tools), where each agent's output feeds into the next. Like an assembly line where each station is an LLM doing one job.
- **Expert agent** = an LLM-driven program prompted to play a specific role — e.g., "you are an Economics expert; you have access to these papers; given this problem, propose how to operationalize the objective." Different from a *real* domain expert; it's a *simulated* expert built from prompts + knowledge base.
- **Coordinator agent** = an LLM whose job is to mediate between expert agents — read each proposal, score it, route it back for revision. Plays the role of a senior researcher refereeing a discussion.
- **One-shot vs. iterative** = "one-shot" means the agent runs once and produces output; "iterative" means it runs in a loop until some convergence criterion is hit.
- **Hydrologic hazard prediction** (Senior X's domain) = predicting things like floods, landslides, drought events. Place-dependent, data-heavy, draws on multiple disciplines (hydrology, statistics, ML) — structurally similar to ride-hailing in those ways.

**The pitch — what you actually want the audience to take away:**

This isn't "ML for ride-hailing" anymore. It's "the lab's preferred research methodology, applied to a different domain." That re-positions the whole proposal as a *methodology transfer* — the same kind of move that makes for clean PhD contributions.

**Calibration note on the advisor quote:** Include it. Don't dwell on it. It signals you know what the lab values without making the proposal *about* the advisor.

**Likely question — "Why is ride-hailing a good domain for this methodology?"**

→ Three reasons:
1. **Genuine method diversity** — the corpus has RL, ILP, MFG, fluid models, structural econ, LLMs. The autoresearch step has something real to pick between.
2. **A good evaluation target** — Castillo's welfare decomposition. Most domains don't have a published peer-reviewed structural target.
3. **Cities clearly differ** — on dimensions that plausibly should affect method choice.

---

## Slide 8 — The Gap & Reframe

**On the slide:**
- **JPDR (TKDE 2024) closes the original gap.** A published RL system already optimizes matching + pricing + rebalancing jointly on NYC data.
- **What JPDR doesn't do — but still matters:**
  - **Context-aware method selection** — JPDR commits to SAC + contextual bandit. The choice of method (vs. ILP, MFG, fluid model) is implicit, not derived from the problem's context.
  - **Welfare-decomposition alignment** — JPDR maximizes scalar revenue. It can't tell you what happened to driver well-being specifically (cf. Castillo: −0.98%).
  - **Strategic-driver heterogeneity** — JPDR treats every driver as identical and obedient. Cook 2021 and Castillo 2025 both document that drivers respond differently by demographic.
- **Reframe:** instead of building a better joint-3 controller, build an *agentic pipeline* that picks and configures the right optimization method given the city — with Castillo's welfare decomposition as the Economics-agent reference (Phase 2) AND the Aggregator's evaluation target (Phase 4).

**The story — four beats:**

1. "JPDR proved joint-3 is feasible. That settles the question 'can we even do this?'"
2. "What's still open is *which method*, *for what city*, *with what welfare incidence*."
3. "Our reframe: stop trying to build a better controller. Build the *agent that builds controllers*."
4. "Castillo shows up twice in the pipeline — as the Economics agent's reference (Phase 2) and as the Aggregator's evaluation target (Phase 4). That preserves the welfare grounding JPDR doesn't have."

**Plain-language version of the three "JPDR doesn't do" bullets:**

- **Context-aware method selection** = JPDR's authors picked their method based on what worked for them on NYC data. Specifically:
  - **SAC** (Soft Actor-Critic — a modern RL algorithm using two neural networks: an *Actor* that picks actions and a *Critic* that scores them, with built-in randomness to keep exploring; well-suited to continuous-action problems like "where exactly should this driver go") — used for dispatching and repositioning.
  - **Contextual bandit** (a simplified RL where at each step you pick one of N options, see the immediate result, and learn — no long-horizon planning; fast to train) — used for pricing.

  There's no argument in the paper that this combination is the *right* method for, say, Tainan or Miri. It might be — but nothing in JPDR derives the choice from the data. A different city might be better served by:
  - **ILP-based matcher** (integer linear programming — math optimization where you make discrete yes/no assignments subject to hard constraints; like solving a giant Sudoku for "which driver gets which rider, given everyone can only take one trip"). Good for small fleets and tight constraints.
  - **MFG-based pricer** (mean-field game — treats the driver population as a continuous "fluid" rather than as individuals, with each driver reacting to the *average* behavior of the rest). Good when there are tens of thousands of drivers and individual differences average out.
  - **Fluid-model controller** (treats the flows of riders and drivers as continuous streams in pipes, used for analytical reasoning about steady-state behavior). Good for high-level pricing/capacity decisions.

  JPDR doesn't open the question of *which* of these is appropriate for *which* city.

- **Welfare-decomposition alignment** = JPDR optimizes one number — GMV (Gross Merchandise Volume, total dollars flowing through the platform). It can't tell you whether a 5% GMV bump came at the expense of drivers (Castillo's −0.98% pattern: riders gain, drivers lose) or with drivers (everyone benefits together). Two very different outcomes that look the same on JPDR's scoreboard.

- **Strategic-driver heterogeneity** = JPDR's model assumes (a) every driver accepts whatever trip the algorithm sends them, (b) every driver responds the same way to the same dispatch, and (c) there's no demographic difference in surge response. Real drivers don't behave that way. Cook 2021 documented the 7% gender earnings gap among Uber drivers; Castillo 2025 showed female and long-hours drivers bear the surge cost disproportionately. JPDR's model can't represent any of that.

**The hardest question you'll get:**

"Is your agent really doing autoresearch, or is it just hyperparameter search dressed up?"

→ Honest answer: that's exactly what Phase 1's pilot is designed to test. If different cities genuinely call for different *method families* (RL for one, ILP for another), it's autoresearch. If they all call for the same method family at different settings, it's tuning. We don't presume the answer.

---

## Slide 9 — Proposed Methodology: Four-Phase Agentic Pipeline (Adapted)

**On the slide:**
- **Phase 1 — City/Geographical Expert** ingests city data + research objective → produces region context schema. *(one-shot)*
- **Phase 2 — Objective Operationalization** (three expert AI agents + a Coordinator, iterative):
  - **Economics agent** — Castillo's decomposition, demographic responses, surge incidence
  - **OR agent** — matching/pricing/rebalancing as constrained optimization, feasibility
  - **ML/RL agent** — state/action/reward space, candidate methods from the corpus
  - **Coordinator** — reviews and scores proposals until they converge
- **Phase 3 — Per-sub-task autoresearch** — given region context + operationalized objective, autoresearch agents pick methods from the 38-paper corpus + JPDR baseline.
- **Phase 4 — Aggregator** — compiles sub-task choices into a deployable controller, evaluated against Castillo's welfare-decomposition target.

**The story:** "This is the architecture. Walk through it left to right. Don't oversell — say what each phase *should* do; the next two slides drill into details. The thing to land hard: the *experts* in Phase 2 are AI agents (LLMs with personas and tools), not actual economists / OR people / ML engineers."

**Terms to know:**
- **OR (Operations Research)** = the discipline of applying math optimization to operational problems — scheduling, routing, assignment, inventory. The OR tradition predates ML by decades and has its own toolkit (linear/integer programming, queuing theory, fluid models).
- **State/action/reward space** = the three things you have to specify before you can do RL. *State* = what the system knows at each moment (driver locations, pending requests, weather). *Action* = what the system can do (assign rider A to driver B, set surge to 1.5×). *Reward* = the score that tells the RL agent how it did (e.g., +1 per completed trip, −0.1 per minute of rider wait).
- **Constrained optimization** = math optimization where you have hard rules that *must* be obeyed (every rider gets at most one driver; surge can't exceed regulatory cap; fleet size is fixed). The OR agent's job is to express the problem in this form so it's solvable.
- **Objective operationalization** = "deciding what to optimize, in concrete math terms." The fuzzy goal ("maximize welfare") becomes a precise function ("maximize Σ rider value − Σ driver disutility − platform cost, subject to constraints C"). Phase 2's three experts iterate on what exactly this function should look like; the Coordinator scores their proposals.
- **Region context schema** = a structured summary of the city — demand patterns, demographics, network shape, regulations — packaged in a way the downstream agents can consume.

**Critical framing — say this out loud the first time:**

The "experts" in Phase 2 are *AI agents*, not real domain experts. Each one is an LLM prompted to play a specific role, with access to a curated knowledge base (almost certainly via RAG — retrieval-augmented generation, see slide 11). Same pattern Senior X's hydrology framework uses. Don't let the audience get confused on this.

**Likely question — "How do the three Phase 2 experts disagree, and how does the Coordinator resolve it?"**

→ That's a Phase 2 design problem (covered on the next slide). The honest answer: "we have a proposal, but we'd want to sandbox it before committing." Convergence criterion is a genuine open question.

---

## Slide 10 — Phase 1 in Detail: The City/Geographical Expert

**What's on the slide:**

- **Input** — raw city data (demand traces, supply data, network topology, regulatory context) + a human-defined research objective.
- **Output** — a *structured region-context schema* fed to Phase 2.
- **Three design choices we're committing to now**, not punting on.

**The story:** "Phase 1 is the smallest piece we can build end-to-end and validate. We're past the open-question stage on the three design choices — here's what we've decided and why."

**Design choice 1 — Schema rigidity: structured JSON with a free-form `notes` field.**

Plain-language version: the City Expert outputs a JSON file with typed sections (demand, supply, geography, regulation) where every field has a known type — *plus* one free-form text field called `notes` for things the schema can't capture. This is the pragmatic compromise: strict where strictness pays off (validation, comparison across cities), flexible where it doesn't.

**Design choice 2 — Missing data: optional fields with confidence flags.**

Most ride-hailing datasets don't include driver demographics (NYC TLC certainly doesn't). Rather than fake the data or give up, every optional field can be `null` with a `confidence` enum: `high` / `medium` / `low` / `unknown`. When something is missing, Phase 2's Economics agent falls back to aggregate proxies (e.g., neighborhood-level census data) and flags the resulting proposal as low-confidence.

**Design choice 3 — Validation: schema-check + sanity bounds (no held-out prediction yet).**

Two layers:
- **JSON Schema check** — every required field present, every type correct, every enum valid.
- **Sanity bounds** — domain rules that filter out nonsense: demand elasticity ∈ [−2, 0], surge cap ≥ 1, fleet size > 0, etc.

We deliberately *don't* do held-out prediction-based validation yet — that's overkill for a proof-of-concept and Phase 2 hasn't been built. We'll add it when there's a Phase 2 to validate against.

**Terms to know:**
- **OD pair (origin-destination pair)** = a recorded trip from point A to point B. "Demand traces" are long lists of OD pairs with timestamps.
- **Network topology** = the structure of the road network as a graph (nodes = intersections, edges = streets). Topology is the underlying map; demand is what flows on it.
- **Demand elasticity** = how responsive demand is to price. If price doubles and demand halves, elasticity is high. Standard econ concept.
- **JSON Schema** = a standard way to write down "what valid JSON looks like" for a given format. You feed your data + the schema into a validator, and it tells you what's missing or wrong.
- **Confidence flag** = a small `high / medium / low / unknown` label attached to a data field that tells downstream consumers how much to trust it.

**Likely question — "Why structured JSON instead of free-form natural language?"**

→ Three reasons: (1) we can validate it programmatically, (2) we can compare schemas across cities mechanically, (3) Phase 2's agents have a known interface to consume. The free-form `notes` field handles the "but cities have weird quirks" worry without breaking the typed interface.

**Likely question — "What if the agent hallucinates a number?"**

→ Sanity bounds catch the egregious cases (negative fleet size, surge cap of 100). For more subtle hallucinations, the next experiment (Phase 1 Simple Experiment, slide 11) is precisely designed to spot-check.

**Likely question — "Why no held-out prediction validation?"**

→ Phase 2 doesn't exist yet — there's nothing downstream to evaluate the schema against. Held-out prediction is the right validation once we have a Phase 2 controller to test, but for the proof-of-concept it would be over-engineering. We'd rather ship the simple version and learn from it.

---

## Slide 11 — Phase 1 Simple Experiment: NYC Taxi Proof-of-Concept

**What's on the slide:**

- **Goal** — show that a prototype City Expert can ingest real ride-hailing data + a research objective and produce a sensible schema.
- **Data slice** — NYC TLC Green Taxi 2016, week 1 of March, Manhattan trips only. ~50–100k records, ~30 MB.
- **The agent** — Claude Sonnet with two tools: `analyze_trips(filter)` for statistics, `validate_schema(json)` for validation.
- **Demo research objective** — "maximize completed trips while keeping median driver hourly income above $20."
- **Success criterion** — the agent outputs a schema-valid, sanity-bounded context with missing fields correctly flagged.
- **Honest scope** — testing the *pipeline*, not the *demand modeling*.

**The story you're telling:**

"We have a plan — so why not test it? The simplest possible experiment we can run is to build a single prototype City Expert, point it at a familiar dataset (the same NYC Green Taxi data JPDR uses), and see whether it can produce a coherent region-context schema. If it works on this slice, Phase 2 has something to consume. If it doesn't, we learn fast and re-plan. Either way the experiment costs us hours, not weeks."

**Why this dataset specifically:**

- **It's the same data family JPDR uses** — anchors our proof-of-concept against the SOTA baseline; if a future Phase 2 produces a controller, we can compare apples to apples.
- **It exercises the missing-data degradation pathway by design** — NYC TLC has zero driver demographics, so the agent has to mark those fields as `unknown` rather than fake them. That's a feature, not a bug — it stresses one of our committed design choices.
- **It's public and tractable** — ~30 MB after filtering. Fast to download, fast to iterate on.

**What the two tools do, in plain English:**

- **`analyze_trips(filter)`** — given a filter spec (date range, borough), returns standard statistics over the trip slice: how many trips per hour, average fare, fare-vs-demand correlation (a proxy for elasticity), unique driver count, OD-pair concentration. The agent calls this to populate the demand/supply/geography sections of the schema.
- **`validate_schema(json)`** — given a candidate JSON object, checks (a) is it valid against the JSON Schema definition, (b) do the values pass sanity bounds. Returns either OK or a list of errors. The agent calls this before declaring its output final.

**Terms to know:**
- **NYC TLC** = New York City Taxi and Limousine Commission. Releases monthly trip-record data publicly. The "Green Taxi" subset is for the outer-borough boro-taxis; "Yellow" is the classic Manhattan medallion fleet.
- **Parquet** = a columnar binary file format, faster and smaller than CSV for tabular data. The NYC TLC data is distributed in Parquet.
- **OD pair (origin-destination pair)** = a trip from point A to point B.
- **Tool use** (in the context of LLM agents) = the LLM doesn't directly compute statistics; it calls a Python function (a "tool") that does the work and returns results, which the LLM then reasons about.
- **Anthropic SDK** = the official Python library for calling Claude. The agent will be built with it.

**Likely question — "Why only one week and only Manhattan?"**

→ Scope. This is a proof-of-concept, not a research result. One week of one borough is enough to (a) populate every field in the schema, (b) exercise the missing-data pathway, (c) sanity-check the agent's output. Scaling to a full month / all boroughs is mechanical once we know the pipeline works.

**Likely question — "What do you expect to find?"**

→ Honest answer: I expect the agent to produce a schema that looks roughly right on the typed fields, with driver demographics correctly flagged as `unknown`, and some hallucination risk in the free-form `notes` field. The point of the experiment is to learn *which* parts are reliable and *which* parts need shoring up before we build Phase 2.

**Likely question — "What's next after this experiment?"**

→ Three things, in order: (1) re-run on a second city (probably Chicago, since they release similar TLC data) to test the cross-city comparability claim; (2) start scaffolding Phase 2 with a single expert agent (Economics) consuming the schema; (3) write up the Phase 1 results as a workshop/short-paper contribution before sinking more time into Phases 2–4.

**Calibration note:** This slide does the "we have a plan, here's the smallest test of it" move. Land that explicitly — most advisors react well to "I'm not just proposing, I'm prototyping."

---

## Slide 12 — Discussion: Why This Fits, Risks, Immediate Next Steps

**Why this fits the lab:**
- **Cross-disciplinary** (CS + Economics + OR) — matches lab values
- **Direct methodology transfer** from the hydrology autoresearch precedent
- **Preserves the 38-paper review and Castillo grounding** as core assets

**Risks worth surfacing now:**
- **The evaluation problem.** The pipeline needs to beat a strong baseline AND we need to argue the *autoresearch step itself* added value — i.e., that we beat "JPDR fine-tuned per city" specifically *because* the agent picked a better method, not just because we threw more compute at it. If a fixed strong baseline matches the pipeline's output, the contribution shifts from "autoresearch wins" to "autoresearch costs more for similar performance" — still publishable, just a different story.
- **Expert-agent grounding.** Each "expert" is only as good as its prompt + knowledge base. Investing in good grounding (RAG corpora, prompt engineering, possibly fine-tuning) is non-trivial work.
- **Scope.** This is a 1.5–2 year project, not a semester one.

**Immediate next steps (paper-sized chunks, parallelizable):**
1. **Pilot Phase 1 on NYC vs. Tainan data** — does the region-context schema actually capture differences that matter? (1–2 months)
2. **Build the method library from the 38-paper corpus** — write down applicability conditions per method. (2–3 months)
3. **Sandbox Phase 2 with two expert agents (Economics + ML/RL)** — does the Coordinator pattern produce sensible objectives? (1–2 months)

**Terms to know:**
- **Baseline** = the existing approach you're trying to beat. In this case, "JPDR fine-tuned per city" is the natural strong baseline because it's the existing SOTA joint-3 RL system.
- **Compute** = computational resources (GPU hours, training time). Throwing more compute at a problem can sometimes get you wins that don't generalize — which is why the *isolation* of the autoresearch contribution matters.
- **Sandbox** = a controlled, low-stakes environment to prototype a system before deploying it. "Sandbox Phase 2 with two agents" = "build a small toy version with just two of the planned agents to see if the pattern works."

**Honest framing for the risks section:** Don't soften these. Naming them yourself signals epistemic honesty and pre-empts the advisor naming them for you. The hardest one — the evaluation problem — is genuinely hard, and pretending otherwise will backfire.

---

## Slide 13 — References

The 10 citations on the slide:
1. **Xu 2018** (Didi RL dispatching) — KDD
2. **Tang 2019** (CVNet) — KDD
3. **Eshkevari 2022** (Didi RL deployment) — KDD
4. **Azagirre 2024** (Lyft RL) — INFORMS J. Applied Analytics
5. **Sun 2022** (JDRL) — KDD
6. **Sun 2024** (JDRCL) — IEEE TKDE
7. **Zhang 2023** (joint pricing + matching) — AAAI
8. **Zhang 2024** (JPDR) — IEEE TKDE
9. **Castillo 2025** (welfare decomposition of surge) — Econometrica
10. **Cook 2021** (gender earnings gap) — Review of Economic Studies

All venues confirmed against primary sources (ACM DL, IEEE Xplore, INFORMS, Econometric Society, Review of Economic Studies).

---

## Things to be ready for that aren't on a slide

**"Is this a thesis or a paper?"**

→ Long-term, a thesis. Each of the three immediate next steps is a paper-sized chunk.

**"How does this differ from AutoML?"**

→ AutoML picks ML methods within a fixed problem class (e.g., "find the best image classifier"). We're picking across problem classes AND operationalizing the objective. Phase 2 has no AutoML analogue.

**"Could you just use JPDR everywhere?"**

→ Maybe. That's literally what the pilot tests. If JPDR-with-per-city-fine-tuning matches the autoresearch pipeline, the contribution shifts to "autoresearch costs more for similar performance" — still publishable, just different.

**"Who's Cook 2021 again?"**

→ Cook, Diamond, Hall, List, Oyer (2021), *Review of Economic Studies*. Decomposed Uber's 7% gender earnings gap among drivers into three factors — experience on the platform, preferences over where/when to work, and driving speed. The gap is *fully attributable* to those three things, *not* to customer discrimination. Empirical companion to Castillo's structural model.

**"What's the relationship to LLM-as-controller papers (LLM-ODDR, GARLIC)?"**

→ Those papers use LLMs *as* the controller — the LLM makes operational decisions (e.g., "assign driver A to rider B right now") in real time. We use LLMs as the *research agent* — the LLM picks and configures the controller, which can itself be non-LLM (RL, ILP, MFG, etc.). Different role, different timescale.

**"How big is the action space for ride-hailing RL? What makes this hard?"**

→ A single city might have ~10,000 idle drivers and ~5,000 pending ride requests at any moment. The naïve action space (all possible assignments) is combinatorial — bigger than any RL agent can directly explore. The standard trick (used by Xu 2018 onwards) is to have RL learn *values* for individual driver-state pairs, then use a combinatorial solver (Hungarian algorithm, ILP) to pick the actual assignment that maximizes total learned value.

**"What's the Hungarian algorithm?"**

→ A classical optimization algorithm (from the 1950s) for the *assignment problem* — given a matrix of costs/values for assigning N workers to N jobs, find the one-to-one matching that minimizes total cost (or maximizes total value) in polynomial time. The workhorse of ride-hailing matching: RL learns the cost matrix, Hungarian finds the optimal assignment.

---

*Last build: deck has been validated against the `endParaRPr`-before-runs schema bug that previously caused PowerPoint to render blank slides. All 13 slides confirmed renderable by LibreOffice; should also render correctly in PowerPoint.*
