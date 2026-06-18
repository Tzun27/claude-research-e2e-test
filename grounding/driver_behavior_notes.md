# Driver Behavior Grounding Notes

Literature grounding for a ride-hailing simulator with a **heterogeneous, strategically-responding driver population**. Covers (A) theoretical OR/economics models of spatial supply response (Besbes 2021, Afèche 2023) and (B) empirical findings on driver behavior (Hall-Horton-Knoepfle 2023; Cook-Diamond-Hall-List-Oyer 2021).

Page numbers for the two repo PDFs are given as `fitz N / printed M` (the PyMuPDF 0-indexed page, and the page number printed on the page). All quotes are verbatim. **Numbers are quoted from source or marked unknown — none are fabricated.**

---

## A. Besbes, Castro, Lobel (2021) — "Surge Pricing and Its Spatial Supply Response" (Management Science)

Repo file: `/home/user/claude-research-e2e-test/Thread_D_Pricing/Besbes2021_Surge_SpatialSupply_MS.pdf`
(working-paper version; printed page = fitz index + 1.)

### A1. How the spatial supply response is modeled
Drivers are a **non-atomic continuum** distributed over a 2D city `C ⊂ R²`. Their spatial response is a **measure-theoretic flow / transport plan** `T` from the initial driver measure `Θ` to an endogenous post-relocation measure.
- "Supply and demand are composed of non-atomic agents... We use non-negative measures to model how these agents are distributed in the city." (fitz 2 / printed 3)
- "The movement of drivers across the city is modeled as a measure on C × C, which we denote by T." (fitz 8 / printed 9)
- Feasible flows preserve mass: `F(Θ) = {T ∈ M(C × C) : T1 = Θ, T2 ≪ Γ}` — first marginal fixed at initial supply, second marginal `T₂` is endogenous post-relocation supply. (fitz 8 / printed 9)
- **No entry/exit (short term):** "drivers respond to pricing and congestion by moving to other locations, but not by entering or exiting the system." (fitz 1 / printed 2)

### A2. How drivers decide WHERE to relocate (equilibrium toward higher earnings)
A **Cournot-Nash / non-atomic congestion (Wardrop-type) equilibrium**: each driver relocates only to the destination maximizing own utility net of travel cost; supply flows to highest-earning locations until no one wishes to deviate.
- Driver payoff: `Π(x, y, p(y), sT(y)) ≜ U(y, p(y), sT(y)) − ‖y − x‖`, with `U(y,...) ≜ α · p(y) · R(y, p(y), sT(y))` — commission share `α` times price times match probability, minus travel distance. (fitz 9 / printed 10)
- Supply equilibrium (Def. 1): `T({(x, y) : Π(x, y, p(y), sT(y)) = ess sup_C Π(x, ·, p(·), sT(·))}) = Θ(C)` i.e. "essentially no driver wishes to unilaterally change his destination." (fitz 9 / printed 10)
- "in equilibrium, drivers only depart for locations that yield the largest earnings." (fitz 6 / printed 7)

### A3. KEY RESULT — local/myopic vs global/network-anticipating pricing
- **Myopic benchmark:** "responding to a shock in demand at the origin by only adjusting the price at the origin; we call this policy the myopic price response." (fitz 22 / printed 23)
- **Magnitude of loss:** the optimal (global) policy's revenue improvement over myopic "can be significant, above 10%" (fitz 29 / printed 30); Table 1 entries peak around **13–14%** (e.g. 13.87).
- **Mechanism / structural insight:** the global-optimal policy *widens the shock's attraction region* and even pushes supply *away* from the periphery to pull *more* drivers toward the shock: "the optimal solution is able to incentivize the movement of a larger mass of drivers towards the demand shock, leading to a mass s^opt(0) = 1.97 versus s^my(0) = 1.66." (fitz 29 / printed 30); "the optimal global price response to a demand shock induces supply movement away from the origin in the inner periphery." (fitz 27 / printed 28)
- No closed-form "price-of-anarchy" constant; loss is quantified numerically (>10%, up to ~13.9%).

### A4. Functional forms (supply as function of price/earnings; flow/balance)
- **Matching / utilization:** `min{1, λ(y)·F̄y(p(y))/sT(y)}` with effective demand `λ(y)·F̄y(p(y))`. (fitz 8 / printed 9)
- **Congestion bound — supply as decreasing function of attainable utility:** `sT(x) ≤ ψ⁻¹_x(V(x| p, T))` Γ-a.e., where `ψx(s) = α·R^loc_x(s)/s` is decreasing. Higher attainable utility `V(x)` ⇒ more post-relocation supply. (fitz 14 / printed 15)
- **Optimal supply within attraction region (Thm 1):** `s_T̂(x) = ψ⁻¹_x(V(z| p, T) − ‖x − z‖)` for `x` near sink `z`. (fitz 18 / printed 19)
- **Flow conservation / no-crossing constraints** in the continuous-knapsack relaxation: `∫ s̃(x)dΓ(x) = Tc`, `s̃(x) ≤ ψ⁻¹_x(V(x))`. (fitz 19–20 / printed 20–21)

### A5. Reservation wage / opportunity cost / earnings equalization
- **No classic reservation wage** (fixed supply, no entry/exit). The outside option is the **pre-shock utility level `ψ1`**: locations not pulled to the shock earn at most `ψ1`: "V(Xl), V(Xr) ≤ ψ1." (fitz 24 / printed 25)
- **Earnings equalization (Wardrop/indifference):** within an attraction region, all drivers obtain equal utility: "in equilibrium, drivers staying or traveling to the origin garner the same utility." (fitz 26 / printed 27); equilibrium utility is a 1-Lipschitz cone around the sink, `V(x) = V(0) − |x|`. (fitz 24 / printed 25)
- Indifference region: `IR(x| p, T) ≜ {y : V(y|p,T) − ‖y − x‖ = V(x|p,T)}`. (fitz 15 / printed 16)
- Explicitly in the Wardrop/selfish-routing tradition: "Our equilibrium concept is similar to the one used by Roughgarden & Tardos (2002)." (fitz 6 / printed 7)

---

## B. Afèche, Liu, Maglaras (2023) — "Ride-Hailing Networks with Strategic Drivers" (M&SOM)

Repo file: `/home/user/claude-research-e2e-test/Thread_D_Pricing/Afeche2023_StrategicDrivers_MSOM.pdf`
(working-paper version dated Feb 2018; printed page = fitz index + 1.)

### B1. How strategic drivers are modeled (decision + objective)
Two-location network. Each driver makes a **participation decision** and a **repositioning decision**, and **maximizes long-run average profit rate** (income per unit time). Drivers *cannot* reject matches.
- "Drivers are self-interested and seek to maximize their profit rate per unit time. They decide whether to join the network, and, if so, whether to reposition to the other network location." (fitz 8 / printed 9)
- Repositioning is a randomization: "A participating driver's repositioning strategy is a vector of probabilities that specify for each network location the fraction of times that the driver will, upon arrival, immediately reposition." (fitz 12 / printed 13)
- Individual objective (Lemma 1, eq. 25): `eπ(η; s, q) = [(γ̄p−c)·T^s(η;s) − c·T^r(η)] / [T^s(η;s) + T^r(η) + T^q(η;s,q)]` — net trip earnings ÷ total cycle time (service + repositioning + queueing). (fitz 17 / printed 18)
- "Drivers cannot reject the platform's matching requests." (fitz 10 / printed 11)

### B2. Admission control / demand rejection to induce repositioning (central result)
- Abstract: "It may be optimal to strategically reject demand at the low-demand location even if drivers are in excess supply, to induce repositioning to the high-demand location." (fitz 0 / printed 1)
- Mechanism: "rejecting rider requests at the low-demand location creates an artificial demand shortage that drivers offset by choosing to reposition more frequently to the high-demand location, rather than joining the low-demand matching queue." (fitz 22 / printed 23)
- Operational lever vs. wages: "By controlling congestion, the platform has an operational lever to incentivize drivers to reposition, as opposed to, for example, increasing their wage." (fitz 22 / printed 23)
- **Proposition 5** gives a necessary-and-sufficient condition (37); holds when cross-location imbalance `Λ12/Λ21` is large, high-demand local share `ρ2` small, relative driving cost `κ` large. (fitz 23 / printed 24)

### B3. Driver equilibrium (where drivers choose to be)
A **symmetric Nash equilibrium in repositioning fractions** with a **spatial indifference / equalized-earnings (Wardrop-like)** structure.
- Equilibrium (eq. 2): `η(λ,ν) ∈ arg max_η eπ(η; λ, w)` — aggregate repositioning equals each driver's best response. (fitz 12 / printed 13)
- Equalized earnings: "Since every driver chooses η(λ,ν), each earns the same profit rate." (fitz 12 / printed 13)
- Indifference when interior repositioning occurs: "drivers are indifferent between repositioning to location 2 and queueing at location 1." (fitz 18 / printed 19) Formally (Prop 2, eq. 28) incentive-compatible iff `q1 ≤ q*_1(s) + k(s)·q2` (equality when `r12 > 0`).
- Paper uses "driver repositioning equilibrium" / Nash; does **not** use the word "Wardrop," but structure is Wardrop-type.

### B4. Reservation wage / opportunity cost / entry-exit + flow equations
- **Heterogeneous reservation wage (idiosyncratic opportunity cost):** "Each driver has an idiosyncratic opportunity cost rate, denoted by c_o ... Drivers join the network if their profit rate per unit time equals or exceeds their outside opportunity cost rate." (fitz 9 / printed 10)
- **Entry / participation equilibrium:** `n = NF(x) = N·P(c_o ≤ x)`; in equilibrium `n = NF(π(λ,ν,n))`. Marginal opportunity cost (reservation wage): `c_o(n) := F⁻¹(n/N)`. (fitz 9, 12, 26 / printed 10, 13, 27)
- **Flow balance:** `λ12 + ν12 = λ21 + ν21`; capacity: busy + repositioning + queueing drivers = participating capacity `n`. (fitz 11 / printed 12)
- Earnings primitive: max profit rate per busy unit time = `γ̄p − c` (revenue rate minus driving cost). (fitz 9 / printed 10)

### B5. Strategic outcome vs. first-best; inefficiency
Three regimes — Centralized (C, first-best), Admission control (A), Minimal/no-control (M). Inefficiency: with decentralized repositioning, drivers reposition only if a **queue** at the low-demand location makes them, wasting capacity.
- "the minimum capacity level required to serve the total offered load, n^M_3, exceeds the corresponding requirement under centralized control, n^C_2, by exactly the size of the queue that will provide the repositioning incentive." (fitz 19 / printed 20)
- Performance ranking (Props 6–7): `ΠM(n) ≤ ΠA(n) ≤ ΠC(n)`. (fitz 25 / printed 26)
- **Caveat:** when strategic demand rejection is invoked, "drivers incur higher driving costs and may be worse off than without admission control (the platform is still better off)." (fitz 25 / printed 26)

---

## C. Cross-cutting theory: reservation wage, entry/exit, spatial equilibrium
- **Reservation wage / entry:** Afèche has it explicitly (heterogeneous `c_o`, participation rule, `c_o(n)=F⁻¹(n/N)`). Besbes does **not** (fixed supply); its "opportunity cost" is the pre-shock utility `ψ1`.
- **Earnings equalization across locations:** both have a **Wardrop/indifference-type spatial equilibrium** — within an attraction region (Besbes) or across repositioning options (Afèche), participating drivers earn the **same** profit/utility, net of travel cost. Besbes: `V(x) = V(sink) − ‖x − sink‖`. Afèche: indifference between queueing and repositioning.

---

## D. EMPIRICAL — Hall, Horton, Knoepfle (2023), "Ride-Sharing Markets Re-Equilibrate"
NBER WP 30883, Feb 2023. Source PDF: <https://www.nber.org/system/files/working_papers/w30883/w30883.pdf>
(page numbers = PyMuPDF index of that PDF)

### Headline: how fast markets re-equilibrate
- **~2 months:** "This market adjustment brings the hourly earnings rate back to about the rate that prevailed before the fare increase, in roughly two months." (Abstract, p.1)
- Dynamic estimate: hourly-earnings effect is "almost precisely 0" in week 0, but "by week 8, the elasticity point estimate is -1, which is close to the estimate of the static effect." (p.27) — i.e. full pass-through to utilization/wait-times by ~8 weeks.

### Driver entry / hours response to a fare increase
- "Following Uber-initiated fare increases, drivers make more money per trip and, initially, more per hour-worked. Drivers begin to work more hours." (Abstract, p.1)
- "with a higher base fare, drivers work more hours in total" (`H*`); "the effects ... on the number of active drivers and the number of hours-worked per active driver. Both increase substantially with a higher fare, though **the intensive margin effect is larger**." (p.23) — i.e. existing drivers working more hours dominates new sign-ups.
- Driver sign-ups: effect "so imprecisely estimated that little can be concluded." (p.23)

### Utilization / "business stealing"
- "this increase in hours-worked—combined with a reduction in demand from a higher fare—has a business stealing effect, with drivers spending a smaller fraction of working hours transporting passengers." (Abstract, p.1)
- Quantified: "a 10% increase in the base fare reduces equilibrium utilization by about 7%." (p.23) Long-run elasticity of utilization w.r.t. fare ≈ **-1** by week 8 (p.27); static estimate ≈ -1.20/-1.40 region (Fig.5 annotations, p.26).
- Long-run hourly earnings pass-through ≈ 0: "There was little to no long-run pass-through of the fare changes into the driver hourly earnings rate, but large changes in wait times and utilization." (p.25)

### Mechanism / model (relevant to simulator)
- Market clearing: `D(p + φ(x)) ≡ x·H((p−c)x)` — passenger demand at full price (fare + wait cost `φ(x)`) equals trips supplied = utilization `x` × hours `H` of the per-hour driver wage `(p−c)x`. (eq.1, p.16)
- Equilibrium utilization = market tightness `x* = D*/H*`; **Prop 1: utilization is decreasing in price**, `dx_eq/dp_eq ≤ 0`. (p.16–17)
- Hours measured "essentially without error"; model "captured with a single supply curve of total hours-worked" — i.e. **upward-sloping, elastic hours-supply curve `H((p−c)x)`** is the engine of re-equilibration. (p.15) Exact elasticity value of `H(·)`: **not reported as a single number** (they note "substantial heterogeneity in individual labor supply elasticities," citing Farber 2015, p.15).

---

## E. EMPIRICAL — Cook, Diamond, Hall, List, Oyer (2021), "The Gender Earnings Gap in the Gig Economy"
Review of Economic Studies 88(5):2210–2238. Source PDF (May 2020 WP): <https://web.stanford.edu/~diamondr/UberPayGap.pdf>; published page: <https://www.restud.com/paper/the-gender-earnings-gap-in-the-gig-economy-evidence-from-over-a-million-rideshare-drivers/>
(page numbers = PyMuPDF index of the WP PDF)

### The gap
- "We document a roughly **7% gender earnings gap** amongst drivers." (Abstract, p.0) — "men make 7% more per hour, on average, for doing the same job in a setting where work assignments are made by a gender-blind algorithm." (p.13)
- Setting has **no gender discrimination** (gender-blind matching/pricing) and **highly flexible hours** — fully explained by behavior, not bias.

### Full decomposition (Gelbach, order-invariant) — three factors
The gap "can be entirely attributed to three factors: experience on the platform (learning-by-doing), preferences and constraints over where to work (driven largely by where drivers live and, to a lesser extent, safety), and preferences for driving speed." (Abstract, p.0)
- **Driving speed: ~half.** "driving speed alone can explain nearly half of the gender pay gap." (p.4) Men drive faster: mean on-trip speed 18.262 vs 17.634 mph (≈0.63 mph faster). (Table 3, p.17)
- **Experience / learning-by-doing: over a third.** "over a third of the gap can be explained by on-the-job learning." (p.4) Gelbach (p.34): "experience explained 50% of the gap"; "Where drivers work can explain a further 28% of the gap." Men accumulate trips faster, so they are more experienced on average.
- **Where they drive / home location: ~10–20%.** "Gender differences in drivers' home location can explain 10-20% of the gap." (p.35) Consistent with spatial-mismatch / commute-cost; safety a smaller secondary factor.
- Robustness decomposition (p.34): "20% of the gender [gap]..." attributable to location features; appendix variant "experience explains about 65 percent ... places of residences ... about ten percent." (p.62)

### Magnitudes for heterogeneity calibration
- **Returns to experience (learning curve):** "drivers who have completed over 2,500 trips make nearly **14% more** than those in their first 100 trips." (p.26) "a fully experienced driver earning about $3 per hour (more than 10%) more than a driver in his or her first 500 trips." (p.24) Learning curve "especially steep early in a driver's tenure," continuing through "at least 2,500 trips." (p.24) Experience binned: 0-100, 100-500, 500-1000, 1000-2500, >2500 trips. (p.17) **No gender difference in the learning process itself** — only in accumulation speed. (p.27)
- Controlling for experience alone shrinks the Chicago gap to "1.4% or roughly a third of the initial earnings gap." (p.26)
- **Free choice of hours / no minimum:** drivers set their own hours; women work fewer hours/week (Chicago/national: men 17.98 vs women 12.82 hrs/wk; trips 31.52 vs 21.83) and have higher attrition (6-month attrition men 65.0% vs women 76.5%). (Table 1, p.11)
- Other earnings-relevant factors favoring men (per-trip, Table 3, p.17): shorter dispatch wait, shorter accept-to-pickup distance, longer trips, higher surge; only incentive pay slightly favors women (~3¢/trip).
- **No effect** from: customer discrimination (Col 2 unchanged), taste for specific hours, or within-week work intensity. (Abstract p.0; pp.16, 27)

---

## F. Implications for the simulator (synthesis)
1. **Repositioning model:** drivers move toward higher *expected net earnings* (gross earnings × match probability − travel/relocation cost), and in equilibrium earnings are *equalized across locations net of travel cost* (Wardrop/indifference — Besbes A2/A5, Afèche B3). A discrete-choice (logit) softening of "go to argmax expected earnings" gives heterogeneity while nesting the Wardrop limit.
2. **Entry/exit via reservation wage:** model a population with heterogeneous opportunity-cost rates `c_o ~ F(·)`; a driver participates iff expected earnings rate ≥ `c_o`. Aggregate participation `n = N·F(expected earnings)` (Afèche B4). This reproduces the empirical re-equilibration: positive earnings shock → entry/more hours → utilization falls → earnings revert in ~2 months (Hall et al. D).
3. **Empirically grounded heterogeneity dimensions:** (i) **engagement/hours & attrition** (men ~18 vs women ~13 hrs/wk; 6-mo attrition 65% vs 77%); (ii) **experience / learning-by-doing** (≈+14% earnings from <100 to >2,500 trips; steep early curve); (iii) **driving speed** (≈0.6 mph spread, explains ~half the gender gap); (iv) **location preferences / home anchoring** (home-location differences explain ~10-20% of the gap; drivers prefer to drive near home). All four are quoted, not assumed.
