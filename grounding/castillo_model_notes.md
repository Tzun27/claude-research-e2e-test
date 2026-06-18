# Castillo (2025) "Who Benefits from Surge Pricing?" — Model Structure & Mechanism Notes

Source: Castillo, J.C. (2025). "Who Benefits from Surge Pricing?" *Econometrica* 93(5):1811–1854.
PDF: `/home/user/claude-research-e2e-test/Thread_D_Pricing/Castillo2025_Surge_Welfare_Econometrica.pdf`

Page numbers below are the **journal page numbers** printed on the page (PDF 0-indexed page = journal page − 1811). All bracketed quotes are verbatim from the PDF (math notation is OCR-approximate; subscripts/Greek symbols sometimes render imperfectly).

**Headline numbers (given, for reference):** Total welfare +2.15% of gross revenue ("or $0.25 per trip"); rider surplus +3.57%; driver surplus −0.98%; platform current profit −0.50%; long-hours drivers and women hurt most. Baseline = reoptimized uniform multiplier (1.174). (p.1812, p.1845)

---

## 0. One-line framing of the whole model

- The model is a **spatial equilibrium** with three parts: "The model is composed of three parts: demand, supply, and a matching technology." (p.1812). It allows "high-resolution spatial and temporal heterogeneity as well as transient random shocks" (p.1812).
- Agents make **long-run** (entry) and **short-run** decisions: "In the long run, they decide whether to enter the market... In the short run, agents who already entered the market make choices using the information they observe in the app." (p.1817–1818).
- Welfare results are "mainly driven by short-run elasticities and by the matching technology", NOT by long-run elasticities (p.1845).

---

## 1. MODEL STRUCTURE — Demand, Supply, Matching

### 1a. DEMAND (Section 3.1, p.1818–1820)
Riders make **two decisions**: (i) open the app & choose a destination (long-run), (ii) after seeing fare + pickup time, request a trip (short-run). Modeled as **static** decision makers: "I model riders as static decision makers... waiting for a lower multiplier is not an important response to surge pricing in the data." (p.1819).

**Trip-request utility (eq. 1, p.1818):**
> "Ui = α(xilh) + β(xi)pi + γ(xi)wi + ϵi" — α = trip value by location/hour/covariates; β(xi)pi = "disutility from paying"; γ(xi)wi = "disutility from waiting"; ϵi error. "She requests a trip if Ui > 0."

- Key params: price coeff β(xi) and pickup-time coeff γ(xi). "The ratio γ(xi)/β(xi) measures the value of time" (p.1818).
- Functional form (p.1833): "The pickup time coefﬁcient is linear, γ(xi) = θwxi. The price coefﬁcient takes the form β(xi) = s(θpxi)"; main spec uses s(x) = −exp(−x) to keep price coeff negative. xi includes constant, log income, occasional-rider dummy, unsurged fare, and dummies for airport/weekend/home (p.1833). Error ηi i.i.d. logistic → "a logit model with nonlinear coefﬁcients" estimated by MLE (p.1833).

**App-opening / arrival (eq. 2, p.1820):**
> "λlhx = AlhxU^ρ_lhx" — arrival rate of riders; Ulhx is the ex ante dollar value of opening the app: "Ulhxi = 𝔼[max{(1/β(xi))Ui, 0}|lhxi]". Distribution of long-run outside option F^u(u) ∝ u^ρ.
- Rider surplus (long-run): "RSlhx = ∫₀^Ulhx (Ulhx − u) dF^u(u) = (Alhx/(1+ρ)) U^{ρ+1}_lhx" (p.1820).
- **Key demand parameter ρ** = long-run demand elasticity (p.1820).

**Estimated demand magnitudes:**
- Average **value of time = $2.25 per minute** (main spec 3); higher for high-income riders, weekdays, long/airport trips (p.1833).
- Average **short-run price elasticity = 0.268** (p.1834). Occasional riders more elastic; high-income frequent riders less elastic; home trips ~half as elastic.
- **Long-run demand elasticity = 0.633** (s.e. 0.059), calibrated from Uber Latin-American discount experiments → "This results in a value of ρ = 179" [i.e. ρ ≈ 1.79] (p.1831, p.1835). Robustness range 0.633–1.5.

### 1b. SUPPLY (Section 3.2, p.1820–1824)
Drivers make **three decisions**: (1) start working (entry); (2) when available, keep working or exit; (3) where to move. Driver j has a discrete **type d** capturing "gender and preferred shift patterns."

**Driver utility (eq. 3, p.1820):**
> "Vj = Σ_t (πjt − ktj)" — sum over worked periods; πjt = net earnings (earnings minus physical driving cost); ktj = per-period opportunity cost of working.
- Crucial homogeneity assumption: "two drivers with an equal opportunity cost who get the same net earnings are equally well off. It does not matter where in the city they drove... whether they were busy or idle." "this is a simplifying assumption that I need for tractability." (p.1820). **This assumption is the reason allocative efficiency does not benefit drivers — see §3.**

**Movement (eq. 5, p.1821):** logit over destinations:
> "Pr(ljt+1 = k|st) = exp(ωlkhd + δd·vdk(st) + ζkt) / Σ_k′ exp(...)" — ω = road/traffic fixed effects (vary by driver type); vdk(st) = mean future earnings if moving to k (eq. 4, an empirical average over next ¯t=45 periods ≈ 90 min, p.1821); δd = sensitivity to earnings. ζkt = unobserved "flocking" shock.
- "Drivers do not respond to surge multipliers directly, but they respond indirectly: higher multipliers lead to higher expected earnings, so δd also measures how movements respond to multipliers." (p.1821–1822).
- Drivers are **boundedly rational** (not fully forward-looking) to dodge the curse of dimensionality: "It is not feasible to model fully rational drivers in this setting because of the curse of dimensionality: the state involves all multipliers surrounding the driver." (p.1822). Identified because "I observe an objective measure of utility—net earnings" (p.1822 fn.19).
- Estimated **δ = 0.0904** (s.e. 0.016) with control function (Table III, p.1837). Interpretation: a $3 earnings rise in one of four surrounding directions (≈ multiplier 1→1.5) raises move probability "from 0.25 to 0.304" (p.1838). One single δ estimated for all types (no power to split by type — p.1836; caveat: heterogeneity in welfare "does not account for additional differences that might arise if some types of drivers are more likely to chase high surge multipliers").

**Exit (eq. 6, p.1822):** stay if expected value of staying > 0.
- Opportunity cost: "kjt = κd max{t − ¯tj, 0} + ςjt" — zero until ¯tj, then rises linearly at rate κd ("drivers get tired or eventually need to get back home"); ¯tj = tj + Tj (start time + driver-specific preferred shift duration Tj).
- **Key exit param κd** = "to what extent drivers respond to high multipliers by staying longer" (p.1823). Estimates (with control fn): full-time men 0.97, part-time men 1.06, full-time women 1.99, part-time women 1.03 (Table III, p.1837).

**Entry (eq. 7, p.1823):**
> "μlhdT = BlhdT·V^{σd}_lhdT" — entry rate; VlhdT = expected utility from working; F^V(V) ∝ V^{σd}.
- Driver surplus: "DSlhdT = (BlhdT/(1+σd)) V^{σd+1}_lhdT" (p.1824).
- **Key entry param σd** = long-run supply elasticity (p.1824). Calibrated (Table III, p.1837): full-time men 0.152, part-time men 0.186, full-time women 0.234, part-time women 0.357.
- Long-run supply elasticities calibrated from Caldwell & Oehlsen (2022) Houston experiment: "0.3 for full-time men, 0.5 for part-time men, 0.6 for full-time women, and 1 for part-time women." (p.1831). Robustness: scaled up by factors 1–2.
- Model "does not allow drivers to start working in response to unexpectedly high multipliers" (p.1824) — argued unimportant.

### 1c. MATCHING TECHNOLOGY (Section 3.3, p.1824)
This is the "main driver of my results" (p.1814).
- "In period t, let It be the set of riders that request a trip and let Jt be the set of drivers that can be offered a trip—those who are available as well as those who will drop off a rider during the next four minutes." (p.1824).
- Pickup time for each rider-driver pair drawn from "a distribution G(·|li, lj, bj, h) that depends on the rider's location li, the driver's location lj, whether the driver is busy bj... and the hour-of-the-week h." (p.1824).
- **Matching is sequential, random order, nearest-driver-first, NOT price-based on the driver side:** "Riders in It are matched sequentially in a random order. For every rider, her trip is offered ﬁrst to the driver in Jt with the lowest pickup time, who accepts it with probability φa if he is available and probability φb if he is busy... until the rider is matched or until no available drivers remain within 10 km, in which case the rider does not get a trip and takes her outside option." (p.1824).
- "A distribution G(·) with higher pickup times means a larger matching inefﬁciency. Lower acceptance probabilities φa and φb result in matches with drivers who are farther away, and, therefore, a larger matching inefﬁciency." (p.1824).
- Estimated acceptance probabilities: "ˆφa = 0.692 (s.e. = 0.036) and ˆφb = 0.138" via simulated method of moments matching mean pickup time (3.52 min) and fraction of trips to busy drivers (13.7%) (p.1840). Acceptance "do not increase with surge multipliers" — drivers see only multiplier + distance, not destination (p.1824 fn.23).
- Surge multipliers generated by "the exact algorithm Uber used"; proprietary, but "it depends on the number of available drivers and on the number of riders who open the app" aggregated over nearby locations and recent minutes (p.1824).

### 1d. EQUILIBRIUM (Section 3.5, p.1825)
- Equilibrium = beliefs consistent with empirical averages; a **fixed point** "xP = f^P(xP)" (eq. 8, p.1825) over beliefs x = (U, V, v, vS).
- "Appendix A proves that an equilibrium exists and that under an additional condition (which is always satisﬁed in simulations), the equilibrium is unique and stable." (p.1825).
- Computed by **simulation** of "the history of the whole market during several weeks, agent by agent", then iterating xn+1 = (1−γn)xn + γn·f̂^P(xn) (eq. 10, p.1826). High-dimensional beliefs (mean future earnings vdk, value of staying vS) computed via dimensionality-reducing regressions (eq. 9, p.1825–1826).

---

## 2. THE "UNIFORM PRICING COUNTERFACTUAL" (the baseline)

- **Definition:** uniform pricing = "a unique multiplier for the whole market" / "prices only depend on trip distance and duration" (p.1812, p.1842). I.e. one surge multiplier applied to all locations and all times of the week. Surge pricing instead is a "mean-preserving spread of surge multipliers" relative to a uniform policy at the same average (p.1842).
- The headline numbers are measured against the **reoptimized uniform multiplier = 1.174**, NOT against the data's average multiplier (1.149) and NOT against profit-maximization. (p.1844–1845)
- **How the overall price level is set (NOT profit-max):** Uber maximizes a **weighted sum** of current profit, rider surplus, and driver surplus (eq. 16, p.1843):
  > "max π + αRRS + αDDS. Uber sets the uniform multiplier that maximizes this objective function."
- Weights are **backed out** to rationalize observed status-quo pricing: "I set αR = 1. I then ﬁnd values of (απ, αD) such that, among the surge pricing policies in Figure 8, the status quo pricing policy maximizes (16)." (p.1843). "all nonnegative solutions lead to almost identical results... the sum of απ and αD is always less than 0.7." (p.1843–1844). So Uber weights rider surplus heavily, profit/driver surplus lightly.
- Optimal uniform multiplier "ranges between 1.168 and 1.178... The average is 1.174, which I hereafter take as the baseline value" (p.1844).
- Uber is **NOT** pricing to maximize current profit: "Driver surplus and proﬁt are increasing for the whole range of multipliers. This means that Uber prices well below the level for current proﬁt maximization" (p.1842). Status-quo overall price is "well below the level that maximizes current proﬁts but close to the point that maximizes rider surplus" (p.1812).
- Key assumption (caveat): Uber "maximizes the same weighted sum in a uniform pricing counterfactual and in the status quo." (p.1813 fn.2).

---

## 3. THE MECHANISM — why welfare↑, rider↑, driver↓, platform↓

The asymmetry "can be decomposed into three parts, each of which arises from the way surge pricing interacts with matching inefﬁciencies." (p.1812). Quotes p.1812–1813:

**(i) Time savings (mitigating supply-demand imbalance):**
> "surge pricing saves people's time as it mitigates imbalances between supply and demand: riders are picked up more quickly, and drivers wait less time between trips. These time savings result in larger welfare gains for riders." (p.1812)
- Why riders gain MORE from time savings despite drivers saving more clock-time: "Drivers save around twice as much time per trip as riders. However, the value of one minute is much higher for riders than for drivers" (p.1842). Rider value of time is high (≈$2.25/min); "The value of time to drivers... is their average hourly earnings, which are slightly above minimum wages." (p.1812–1813). (Note: surge saves only ~7 sec/trip for riders on average; "the main effect is that surge pricing cuts down the upper tail of the pickup time distribution" — p.1841 fn.49.)

**(ii) Allocative efficiency — benefits riders ONLY:**
> "When drivers are scarce, surge pricing assigns trips to riders who have a high willingness to pay, rather than randomly (as uniform pricing does). But there is no such increase in driver surplus." (p.1813)
- WHY drivers get no allocative gain: drivers are nearly indifferent across trips because (a) the homogeneous-utility assumption (eq. 3) and (b) information: "Drivers observe little information about trips before accepting them—only the surge multiplier and the distance to the rider." (p.1813). Also: "Surge pricing has no effect on the matching process, so it does not help reallocate trips towards drivers who are closer." (p.1843); "The value of getting a trip (relative to being idle) only varies across available drivers with their distance to the rider" (p.1843).

**(iii) Lower average prices under surge → driver↓ and platform↓:**
> "prices are lower on average with surge pricing, decreasing driver surplus and the platform's current proﬁts." (p.1813)
- The reoptimized uniform multiplier (1.174) is **higher** than the average surge multiplier, so moving to surge lowers average prices, transferring surplus from drivers/platform to riders. Net: "the decrease from lower prices outweighs the small increase from time savings" for drivers and profit (p.1845).

**WHY does the PLATFORM's profit go DOWN under surge (the counterintuitive part)?**
- It is NOT that surge is bad for Uber's profit at a given price level — at a fixed average multiplier, surge pricing RAISES profit slightly (Figure 8; "Driver surplus and proﬁt are also higher for surge pricing, but only slightly", p.1842; effects on profit "are almost exactly proportional to the effects on driver surplus", p.1841 fn.48).
- The −0.50% arises because **the optimal uniform price is higher than the surge average**, so the comparison bundles a price-level cut with the switch to surge. WHY is optimal uniform pricing higher? The **wild-goose-chase** mechanism (Castillo, Knoepfle & Weyl 2023):
  > "for a given time and place, it is worse to err by setting prices too low than too high, in part because of a matching failure... that hurts all market participants when drivers are scarce. Under uniform pricing, the only way to avoid that failure is with a high multiplier at all times. With surge pricing, the platform can set a lower average price, and surge pricing automatically increases prices during high demand times, avoiding the problem." (p.1813)
  > "Driver scarcity is bad for riders because they are matched to far-away drivers. But it is also bad for drivers, who must spend a long time picking up riders... A negative feedback loop begins: driver scarcity leads to inefﬁcient driver time use, further fueling driver scarcity... a situation that Castillo, Knoepﬂe, and Weyl call a wild-goose chase." (p.1844)
- Interpretation of the profit drop: "This probably does not mean that Uber is hurt, but rather that it is willing to forgo current proﬁts to increase rider surplus" (p.1812), maximizing long-run shareholder value via customer retention. Profit is computed "on a time frame of a few weeks" (p.1842).

**Net welfare ↑ (+2.15%):** "Surge pricing dominates uniform pricing in the sense that, for every level of the average multiplier, welfare is higher with surge pricing." (p.1842). Welfare has "an inverted-U shape with a maximum at around 1.4. Uber, therefore, is pricing lower than is socially optimal." (p.1842).

> NOTE on magnitude: "Welfare effects are somewhat small because surge pricing only affects 21.2% of trips (i.e., the surge multiplier is above 1). For the remaining 78.8% of trips, the multiplier is equal to 1." (p.1812 fn.1).

---

## 4. WHY LONG-HOURS DRIVERS (and women) ARE HURT MOST

Mechanism is about **WHEN** drivers work relative to when multipliers are high — NOT about elasticity differences. Quotes p.1847:
- "The heterogeneity in this ﬁgure is mainly driven by the fact that certain types of drivers are more likely to work when surge pricing is high (Section 5.2.3)." (p.1847)
- "Full-time drivers are hurt the most by surge pricing: for every hour of work, the surplus of full-time women goes down by $0.13, and the surplus of full-time men decreases by $0.08." (p.1847)
- WHY: "This is so because they tend not to work during bar hours, when surge multipliers are highest (see Figure 2). Full-time men partially compensate by working during the early morning, when surge multipliers are also high, but full-time women do not—hence, they are hurt the most." (p.1847)
- Part-time drivers ≈ indifferent on average, but heterogeneous: those working bar hours / early morning gain $0.15–0.3/hr; those working weekday morning/midday (low multipliers) lose ~$0.1/hr (p.1847).
- The elasticity channel is explicitly **second-order**: "Driver types also differ in terms of how responsive their entry and exit decisions are to expected earnings (Table III). However, that only has a minor (second-order) effect on driver surplus since it only affects marginal drivers who are close to indifferent" (p.1847 fn.51).
- Driver classification: 7 k-means clusters; "full-time drivers" = 13.4% of drivers, 36.8% of hours, highly experienced, work day/evening; rest "part-time" (p.1817). Women = 22.1% of drivers, 15.6% of hours (p.1817).

---

## 5. SPATIAL / TEMPORAL STRUCTURE (Section 2, p.1815–1816)

- City: Uber Houston, **March 16 – April 8, 2017**; UberX only; Lyft absent → "a clean setting... without having to consider competition between platforms." (p.1815).
- Region: center of Greater Houston, "area is 203 km2 and its population is 520,000" / "1.3 times the area of Washington, D.C." (p.1812, p.1815). Accounts for "53.2% of all trips, but for 85.7% of trips with a surge multiplier above 1." (p.1815).
- **Spatial discretization: 1681 hexagons ("locations", index l ∈ L), each ≈ 400 m across.** "There are 1681 hexagons in the region of analysis; each one is roughly 400 meters across." (p.1815).
- **Temporal discretization: 2-minute periods (index t ∈ T)** — "the level at which surge pricing varies"; multipliers "simultaneously updated every two minutes" (p.1815, p.1818). Hour-of-the-week h also used for fixed effects (168 hours/week).
- Surge multiplier varies spatially by hexagon and temporally by 2-min period; "depends on the origin of the trip" (p.1812). Average multiplier 1.149; 78.8% of riders see multiplier = 1 (p.1818). "Less than a third of the variation in surge multipliers is predictable" (R² = 0.284) (p.1818).
- Fare structure: base fare = "$3.30 + $0.87 × (predicted miles) + $0.11 × (predicted minutes)", times a real-time surge multiplier; final price "pi = b + mlt(¯pi − b)" with b = $2.30 booking fee (p.1818, p.1827). Uber commission 24–28% (avg 26.3%); booking fee $2.30/trip (p.1818).

---

## 6. WELFARE ACCOUNTING — how RS, DS, profit are defined/computed (Section 6.1, p.1840)

- **Rider surplus:** "RS = Σ_{lhx} RSlhx" — sum over locations, hours-of-week, rider types of the long-run rider surplus from §3.1.2 (eq. 14, p.1840). RSlhx = (Alhx/(1+ρ)) U^{ρ+1}_lhx, measured relative to the long-run outside option.
- **Driver surplus:** "DS = Σ_{lhdT} DSlhdT" — sum over locations, hours, types d, shift-durations T of the entry surplus from §3.2.3 (eq. 14, p.1840). DSlhdT = (BlhdT/(1+σd)) V^{σd+1}_lhdT, relative to the outside option of not working.
- **Platform (current) profit (eq. 15, p.1840–1841):**
  > "π = Σ_n ((1 − τ − ν)pn − yn − In)" — n indexes trips; pn = trip price; yn = driver payment; In = insurance cost; τ = 2% sales tax; ν = 1% credit-card cost. Commission 26.3%; insurance $0.30/mile when picking up/dropping off (p.1841 fn.42–43).
- **Total welfare** = "the sum of rider surplus, driver surplus, proﬁt, and tax revenue" (p.1841). Tax revenue is a "very small fraction" and not plotted (p.1842 fn.45).
- Driving cost assumed $0.26/mile (fuel, maintenance, repairs, depreciation) for net earnings (p.1825 fn.24).
- All headline % are "of gross revenue" and measured **relative to surge pricing** in the figures, then re-signed for the surge-vs-uniform comparison (p.1845, Figure 11).

---

## 7. DRIVER REPOSITIONING / SPATIAL SUPPLY RESPONSE TO SURGE

- Surge induces repositioning **indirectly via expected earnings**, governed by δd in the movement logit (eq. 5). "Drivers do not respond to surge multipliers directly, but they respond indirectly: higher multipliers lead to higher expected earnings, so δd also measures how movements respond to multipliers." (p.1821–1822).
- **Market-clearing description (Section 3.5.2, p.1826)** — the clearest statement of spatial supply response:
  > "Suppose that one day, by pure chance... too many riders request trips from some neighborhood and deplete it of available drivers. Prices and pickup times go up, so fewer riders request trips. Drivers infer from high multipliers that earnings in that neighborhood will be high—trips will be more expensive, and drivers are scarce so they will be matched quickly—so they move in to the area and work longer. Riders' and drivers' responses, thus, rebalance the market." (p.1826)
  > (Steady-state version:) "If there were, on average, a scarcity of drivers downtown during rush hour, surge multipliers and pickup times would both be high on average... more drivers would start working, move downtown, and work longer if they are downtown. All these forces counteract the scarcity of drivers." (p.1826)
- Magnitude is **modest**: estimated δ = 0.0904; a $3 directional earnings increase (≈ multiplier 1→1.5) raises the probability of moving that direction only "from 0.25 to 0.304" (p.1838).
- Movement is **boundedly rational / not fully forward-looking**; drivers respond to mean future earnings over the next ¯t=45 periods (~90 min), via a low-dimensional regression of earnings on state (p.1821–1822, p.1825–1826).
- **Caveat the model imposes:** a single δ for all driver types (no power to estimate by type), so welfare heterogeneity "does not account for additional differences that might arise if some types of drivers are more likely to chase high surge multipliers." (p.1836). Drivers outside the region of analysis: "I assume that their movements when they are far from that region are unchanged in counterfactuals." (p.1825). 94% of drivers start working outside the area and drive in (p.1824 fn.21).

---

## 8. THINGS NOT IN THE PAPER / EXPLICIT LIMITATIONS (do not over-claim)

- **No deep RL / no learned controller.** The paper takes "the design of the algorithm as a given" (Uber's actual proprietary surge algorithm) and does NOT design or learn a pricing controller (p.1814). Counterfactuals scale the existing multiplier or impose a uniform multiplier (p.1841–1842). (Relevant: the thesis's RL controller is the *new* contribution; this paper supplies the welfare-incidence target, not an RL method.)
- **No destination-based pricing / no driver foresight of destination:** "surge pricing could bring additional efﬁciency gains if drivers saw destinations before accepting trips... but that is not the case in my data, so I cannot quantify this potential beneﬁt." (p.1813).
- **No competition / single platform** (Houston, Uber-only); external validity to multi-platform or high-transit/high-density cities is limited (p.1813–1814).
- **No income effects** in driver labor supply (p.1820 fn.18). No unobserved heterogeneity in β(xi), γ(xi) (p.1833).
- Long-run elasticities are **calibrated, not estimated** from this market's exogenous variation (p.1812, p.1831); robustness in Table IV (p.1846) and Appendix E.
- Suburbs / low-density areas excluded (p.1814).
