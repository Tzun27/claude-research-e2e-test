# Simulator Calibration Notes — Spatial Ride-Hailing / MoD RL Literature

Grounding for a tractable spatial ride-hailing / mobility-on-demand simulator for RL experiments.
Quotes are verbatim; page numbers are 0-indexed PDF pages (as extracted via PyMuPDF). "Unknown" = not stated in the paper.

Papers:
- **G21** = Gammelli et al. 2021, "Graph Neural Network RL for AMoD Systems" (CDC). `Thread_C_AMoD_Hybrid_Learning_Optimization/Gammelli2021_GNN_RL_AMoD_CDC.pdf`
- **L18** = Lin et al. 2018, "Efficient Large-Scale Fleet Management via Multi-Agent Deep RL" (KDD). `Thread_B_GraphMARL_MeanField/Lin2018_LargeScale_FleetMgmt_MARL_KDD.pdf`
- **X18** = Xu et al. 2018, "Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms" (Didi, KDD). `Thread_A_Production_Offline_RL_Dispatching/Xu2018_LargeScale_OrderDispatch_Didi_KDD.pdf`
- **Z23** = Zhang et al. 2023, "Future Aware Pricing and Matching for Sustainable On-Demand Ride Pooling" (AAAI). `Thread_D_Pricing/Zhang2023_FutureAware_Pricing_Matching_AAAI.pdf`
- **E22** = Eshkevari et al. 2022, "Reinforcement Learning in the Wild" (Didi, KDD). `Thread_A_Production_Offline_RL_Dispatching/Eshkevari2022_RLinTheWild_Didi_KDD.pdf`

---

## 1. SPATIAL discretization (grid size, hex vs square, #zones, cell size in km)

**G21 (square grids; primary "tractable" reference):**
- Main experiments use a **16-dimensional, i.e. 4×4 grid** over both cities. "we study system per­formance when deﬁning a 16-dimensional (i.e. 4 × 4) grid over the cities of Chengdu, China and New York, USA." (p.5–6)
- Service areas: "Manhattan (8 a.m. – 10 a.m., with a size of 4 km×4 km) and Chengdu (7 a.m. – 10 a.m., with a size of 10 km×10 km)" (p.5). Areas "divided into grid-like blocks, each of which represents a station." (p.5)
- Cell size: "Each grid cell deﬁnes an area of approximately 6 km2 and 1 km2 for Chengdu and New York, respectively." (p.6) — i.e. NYC cells ~1 km², Chengdu cells ~6 km².
- Scaling experiments go to **8×8 (64 stations)**, and granularities "ranging from 4 × 4 to 20 × 20 grids (with 2 × 2 increments)" (p.7); computational study spans "16 up until 400 stations" (p.7).
- Graph: "transportation network where each station is connected to its spatially adjacent regions" (p.4). Nodes = stations/areas; complete graph G=(V,E).

**L18 (hexagonal grid):**
- "we use hexagonal-grid world to represent the map" (p.2). Orders "served by the available vehicles in the same grid or six nearby grids" (p.2) — hex => 6 neighbors.
- Real data region: "The city is covered by a hexagonal grids world consisting of **504 grids**." (Chengdu, Didi data) (p.4)
- Inter-grid distance: "the travel distance from one grid to another is approximately **1.2km**" (p.6).

**X18 (square grid in toy; region-quantized in production):**
- Toy example: "orders and drivers operating in a simple map of **9 × 9 spatial grids** with 20 time steps." Movement: "drivers are restricted to either stay or move vertically/horizontally by one grid" (square/Manhattan) (p.5).
- Production: state quantized into regions, "s = (t,д) ... t ∈T is the time index and д ∈G is the region's index"; |S| = |T|×|G| (p.2). Exact #regions / cell size **unknown** (production hexagon mention is in prompt, not explicit in this PDF; X18 uses generic "region" g).

**Z23 (road-network graph, not a grid):**
- Uses **street intersections as locations**, not cells: "The resulting network has **4373 locations (street intersections) and 9540 edges**." (Manhattan, OSMnx 'drive') (p.6)
- "we only consider the street network of Manhattan as a majority (∼75%) of requests have both pickup and drop-off locations within it." (p.6)

**E22:** Spatial value function is grid/coordinate-based ("encoded version of the current location of the agent (driver)", one-hot φ_s, p.2; "grid-based smoothed price", p.2). Uses hexbins ("availability of hexbins", p.7). Exact grid size/cell km **unknown** (multiple international cities, anonymized).

---

## 2. TEMPORAL discretization (step length, episode length, #steps/episode)

**G21:**
- Time horizon discretized into intervals T={1..T} of length ΔT (p.3). Poisson "rates are aggregated from the trip record data every **15 minutes**." (p.5)
- Training: "episode length **T = 60**" with "discount factor 0.97" (p.5). (Episode = morning commute window; NYC 8–10am, Chengdu 7–10am.)

**L18:**
- "split the duration of **one day** into **T = 144 time intervals (one for 10 minutes)**." (p.2)
- Episode = 144 steps: "GMV ... over one episode (144 time steps in the simulator)" (p.5). Time-step = **10 min**.

**X18:**
- Production time slots: dispatch rounds "say one or two seconds" (p.1) / "say 2 seconds each" (p.5) for matching; value-function time granularity is **minutes** ("time granularity in the value functions (in minutes)", p.4). Example uses "10-minute windows" (p.2).
- Episode: finite-horizon MDP, "an episode records the transactions of a driver in a **day**." (p.2)
- Toy: **20 time steps** (p.5).

**Z23:**
- Batching / decision epoch: "these requests are batched together over a fixed interval (e.g., **60 seconds**)" (p.1); "The decision epoch duration is set as **60 seconds**." (p.7)
- Episode: "We evaluate the approaches over **24 hours** ... starting at midnight" (p.6) => 1440 one-minute epochs/day.

**E22:**
- Matching batch / dispatch round: "batched matching is repeated every **two seconds**" (p.5); low-latency module "computes dispatch recommendations in every two seconds" (p.7).
- Value-update module frequency: "once every **10 seconds**" (= "five rounds of dispatch") (p.4, p.7).
- Episode/eval: simulation "over three days" per city (p.5).

---

## 3. DEMAND (generation, arrival rates, patterns)

**G21:**
- "the arrival process of passengers for each OD pair is a **time-dependent Poisson process** ... independent of the arrival processes of other OD pairs" (p.3); "estimated from real trip travel data" (p.4). Rates aggregated every 15 min (p.5).
- Trip (i→j) characterized by demand d_ij^t and price p_ij^t (p.3).
- Data: NYC TLC March 2013; Didi Chuxing Gaia (Chengdu) Nov 2016 (p.5).
- Served demand magnitudes (per episode, oracle): Chengdu 4×4 ~42,425 trips; NYC 4×4 ~8,865 (Tables I/II, p.5–6) — note NYC much lower in this morning-window setup.

**L18:**
- "the orders emerge stochastically in each grid" (p.2). Generation by bootstrap: "The new orders generated at the current time step are **bootstrapped from real orders** occurred in the same time interval." (p.5)
- Order has price, origin, destination, duration (p.4). Demand-supply gap analyzed (e.g. airport spikes after midnight, p.8).

**X18:**
- Production: replays real orders ("rewind real-happened orders in this date as demands", p.6).
- Toy demand pattern (explicit, useful template): "simulate realistic traffic patterns with a **morning-peak and a night-peak**, centralized on different locations of residential areas and working areas." Starting locations from a **two-component Gaussian mixture** (p.5):
  - π=(1/3, 2/3); μ1=[3,3,5], μ2=[6,6,15]; σ1=σ2=[2,2,3] (dims = x, y, time) (p.5).
  - Destinations & driver init: uniform over grids (p.5).

**Z23:**
- Demand from **NYC Yellow Taxi 2016**: "on an average **322714 requests in a day** (on weekdays) and **19820 requests during peak hour**." (p.6) => ~224 req/min daily avg, ~330 req/min at peak.
- "the number of requests per minute is around **300**" (peak, used as a 300-dim pricing search space) (p.4).

**E22:**
- Real DiDi requests replayed; demand volume not given as rate but call rate: low-latency module "∼**200 calls per seconds**" averaged; matching-graph edges "can exceed **800 per second** in peak hours" (p.7).
- Mean trip lengths per city ~4.3–6.4 km (Table 1, p.5).

---

## 4. SUPPLY (#vehicles, movement / travel-time model, speed)

**G21:**
- M single-occupancy vehicles; idle availability m_i^t ∈[0,M] per station (p.3–4).
- **Travel time = integer number of time steps** along shortest path: "The travel time for edge (i, j) ... the number of time steps it takes a vehicle to travel along the shortest path ... an integer τij ∈Z+." (p.3). Passenger departing i at t arrives j at t+τij (p.3).
- Travel times "are given and independent of the control of the AMoD ﬂeet" (no endogenous congestion) (p.3). Fleet size M: exact number **unknown** (not stated numerically).

**L18:**
- "On average, the simulator has **5356 agents per time step** waiting for management." (p.5)
- Movement deterministic, one grid per step: "the action is deterministic: if a_t^i ≜[g0, g1], then agent i will arrive at the grid g1 at time t + 1." (p.2) => effective speed ≈ 1.2 km / 10 min ≈ **7.2 km/h** (one hex hop per step).
- Drivers go online/offline stochastically (MLE-fit from data) (p.5). Robustness tested at 100%/90%/10% of initial vehicles (Table 1, p.7).

**X18:**
- Toy: 100 orders with **25 / 50 / 75 drivers** (order-driver ratios 4:1, 2:1, 4:3) (p.5). Drivers move 1 grid/step or stay (p.5).
- Travel/pickup time via **ETA**; state advance s→s'' over Δt = pickup + wait + delivery time steps; idle Δt=1 (p.2–3). Production fleet size **unknown**.

**Z23:**
- Fleet sizes swept: **1000 / 1500 / 2000 vehicles** (Table 4, p.6); income-equivalence study uses 530–1380 vehicles (Table 6, p.7).
- Vehicle **capacity c (ride-pooling)**: swept **2 and 4** (Tables 4–5). Quality constraints: pickup-delay τ, detour-delay λ (Table 1, p.1; Ft^v def p.5).
- Travel times: "daily mean travel time estimate" per road segment (Santi et al. 2014) (p.6).

**E22:**
- Fleet size **unknown** (production, multiple cities). Movement implied by OD matrix; pickup distance enters as penalty term f(s,s') (p.3).

---

## 5. MATCHING (radius, batch window, method, pickup constraints)

**G21:**
- Matching solved as an **LP/assignment** maximizing profit: max Σ x_ij(p_ij − c_ij) s.t. 0≤x_ij≤d_ij (Eq.2, p.3). Constraint matrix totally unimodular => integer flows. Solved with **CPLEX** (p.5).
- Three-step framework: (1) passenger matching, (2) policy picks desired idle distribution, (3) min-cost rebalancing (p.3). Unmatched passengers leave the system (p.3). No explicit pickup radius (matching is per-OD-pair flow).

**L18:**
- **Two-stage greedy by grid neighborhood**: "In the first stage, the orders in one grid are assigned to the vehicles in the same grid. In the second stage, the remaining unfilled orders are assigned to the vehicles in its neighboring grids." (p.5)
- Matching radius = same grid + 6 adjacent hexes (p.2).

**X18:**
- **Bipartite matching solved exactly with Kuhn-Munkres / Hungarian (KM)**: "we employ the Kuhn-Munkres (KM) algorithm to solve it." (p.4). Edge weight = Q-value (advantage) of driver-order pair.
- Batch window: "during each short time slot (say one or two seconds), the platform's decision center first collects all the available drivers and active orders" (p.1).
- Toy pickup constraint: "orders can only be dispatched to drivers in **Manhattan distance ... no greater than 2**." Cancellation: "truncated Gaussian in the range 0 to 5 with mean 2.5 and std 2" time steps (p.5).

**Z23 (ride-pooling => tripartite):**
- **Maximum-weight matching on a tripartite graph (requests–trips–vehicles), solved as an ILP** (CPLEX 20.1) (p.2, p.6). Each vehicle ≤1 trip; each request ≤1 vehicle (Eq.12, p.5).
- Trip feasibility (Alonso-Mora 2017): pickup delay ≤ τ and detour delay ≤ λ (p.5). Batch window 60 s (p.1).

**E22:**
- **Maximum bipartite matching via Hungarian algorithm**, batched every 2 s (p.2–3, p.5).
- **Adaptive graph pruning (LM-UCB)**: drop driver-order edges whose completion probability p_c ≤ threshold th; threshold tuned online by a multi-arm bandit on CR+0.1·AR (p.3–4). Penalty term f(s,s') = pickup distance/wait time (p.3).

---

## 6. PRICING (fare model, surge range, acceptance/elasticity) — esp. Z23

**Z23 (the pricing reference):**
- **Base price + multiplier**: a base price µ_0^r is computed per request "based on factors such as source, destination, travel distance, etc." (p.4). Candidate price = base × multiplier: "˜µ_v^r = µ_0^r · a_v" (p.4).
- **Surge / price-factor action set (discrete)**: "Av = {0.8, 0.9, 1.0, 1.1, 1.2}" (p.4). => multiplier range **0.8–1.2** in 0.1 steps.
- **Acceptance / price-sensitivity = logistic (sigmoid) in price ratio**, fit to Uber data (Yan et al. 2019):
  > p_r(µ_v^r) = 1 / ( 1 + e^{ 0.67·(µ_v^r/µ_0^r) − 1.69 } )   (Eq.15, p.6)
  - At µ=µ_0 (ratio 1): exponent = 0.67−1.69 = −1.02 => p_accept ≈ 0.735.
  - At ratio 1.2: exponent = 0.804−1.69 = −0.886 => p ≈ 0.708; at ratio 0.8: −1.154 => p ≈ 0.760. (Acceptance falls as price rises, as expected.)
- **More price-conscious markets** modeled by "scaling the coefficient of the final price and the constant term in Equation 15 by a factor of 10 which makes the transition from 'accept' to 'not accept' more dramatic." (p.7) => sharper elasticity for developing-market scenarios.
- Reward in expectation = Σ_r µ_v^r · p_r(µ_v^r) (p.4–5). Profit objective: o = Σ_r p_r(µ_r)·µ_r − cost_f^v (p.2).
- Pricing learned via **Mean Field Q-Learning**; Boltzmann/softmax action selection (Eq.4, p.4).

**G21:** Prices p_ij are exogenous, "externally decided and known beforehand", from trip-record data; rebalancing trips have p=0 (p.4). No surge / elasticity (demand independent of price). Profit per matched trip = p_ij − c_ij (p.3).

**X18:** Reward = **order price** (=GMV); discounted across trip duration: R_γ = Σ_{t=0}^{T−1} γ^t (R/T) (Eq.1, p.2). No surge/elasticity modeled; price taken from data/ETA. Example: $30 trip, 20-min ride, γ=0.9 (p.2).

**L18:** Order value from data ("An order with value 10", Fig.1, p.2). No pricing decision / elasticity.

**E22:** Price (fare r_ss') is exogenous and treated as a **noisy signal smoothed** (rolling average / momentum, Eq.7) to decouple dynamic-pricing & marketing incentives from value learning (p.2). No surge decision in this paper (dynamic pricing handled by separate systems).

---

## 7. ECONOMICS (commission/take rate, fare per km/min, costs, earnings)

**G21:**
- Trip profit = revenue − cost: reward r = Σ x_ij(p_ij − c_ij) − Σ y_ij c_ij (Eq.5, p.4). Cost c_ij is a function of travel time τ_ij (p.3).
- Rebalancing cost in $ reported (e.g. NYC 4×4 oracle ~$4,647/episode; Chengdu ~$1,453) (Tables I/II). Exact fare-per-km / commission **unknown** (derived from trip records).

**L18:**
- Objective = **GMV** (value of all served orders) (p.2). Reward per agent = averaged order value at destination grid (p.2).
- **Reposition cost (fuel)**: "the travel distance from one grid to another is approximately 1.2km and the fuel cost is around **0.5 RMB/km**, we set the cost of each reposition as **c = 0.6**." (RMB) (p.6). Episode GMV = served-order value − total reposition cost (p.6).
- Commission/take rate **unknown** (GMV, not platform net).

**X18:**
- Objective = **GMV** (sum of order prices) (p.2). Reward = order price, discounted by trip duration (Eq.1). Production revenue gains 0.5–5% reported (p.1). Commission rate / fare-per-km **unknown**.

**Z23:**
- Objective = revenue/profit; cost term cost_f^v = "marginal increase in cost incurred by serving a trip f with vehicle v" (p.2). Distance/fuel proxied by km-travelled (sustainability metric, Table 7: ~9–11 km/vehicle/hour) (p.7).
- Fare per km / commission **unknown** (base price from distance+time, not a stated rate).

**E22:**
- Objective = **total driver income** (∝ monetary value of completed trips) + CR + AR (p.4–5). Fare r_ss' = "the fare for the driver locating at state s to pick up a rider and drop them off at state s'." (p.2). Commission/take rate **unknown**.

---

## Cross-paper quick-reference table

| Dim | G21 (AMoD) | L18 (MARL fleet) | X18 (Didi dispatch) | Z23 (pricing+pool) | E22 (Didi deployed) |
|---|---|---|---|---|---|
| Spatial | 4×4 sq (up to 20×20); NYC 1 km²/cell, Chengdu 6 km²/cell | hex, 504 grids, 1.2 km hop | 9×9 toy sq; region-quantized prod | road graph 4373 nodes/9540 edges (Manhattan) | grid/hexbin (size n/s) |
| Time-step | ΔT, demand binned 15 min; T=60/episode | **10 min**, 144/day | ~2 s match; value gran. min; toy 20 steps | **60 s** epoch, 24 h/day | **2 s** match, 10 s value update |
| Episode | morning window, 60 steps | 1 day (144) | 1 day | 1 day (1440) | ~3 days |
| Demand | time-dep Poisson from data | bootstrap real orders | replay; toy 2-Gaussian peaks | replay NYC; 322k/day, ~300/min peak | replay; 200 calls/s |
| Supply | M veh (n/s); τ_ij integer steps | **5356/step**; det. 1 hop/step | 25/50/75 toy; ETA | **1000/1500/2000**; cap 2 & 4 | n/s |
| Matching | LP profit-max (CPLEX) | 2-stage greedy (same+6 nbr) | **Hungarian/KM** bipartite | **ILP tripartite** (CPLEX) | **Hungarian** + LM-UCB pruning |
| Pickup constraint | none (OD flow) | same+adjacent hex | Manhattan dist ≤2 (toy) | pickup-delay τ, detour λ | pickup-dist penalty + prob threshold |
| Pricing/surge | exogenous, p=0 rebal | order value from data | order price (data) | **base × {0.8–1.2}**, MFQL | exogenous, smoothed |
| Elasticity | none | none | none | **logistic** Eq.15 (Yan 2019) | none |
| Economics | profit p−c; cost f(τ) | **GMV − 0.6 RMB/reposition** | GMV (order price) | revenue − cost_f; km metric | driver income; fare r_ss' |
| Commission | n/s | n/s | n/s | n/s | n/s |

n/s = not stated / unknown.
