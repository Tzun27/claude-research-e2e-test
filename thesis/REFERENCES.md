# References

Bibliography for the thesis. Calibration sources (independent of Castillo's
welfare numbers) are marked **[calib]**; corpus papers from the repo's literature
review are marked **[corpus]**.

## Economics / labor / pricing

- **[calib]** Cohen, P., Hahn, R., Hall, J., Levitt, S., & Metcalfe, R. (2016). *Using Big Data to Estimate Consumer Surplus: The Case of Uber.* NBER Working Paper 22627. — Demand price elasticity ≈ −0.55 (most estimates −0.4 to −0.6) via regression discontinuity at rounded surge thresholds; ~$1.60 consumer surplus per $1 spent.
- **[calib]** Chen, M. K., Chevalier, J. A., Rossi, P. E., & Oehlsen, E. (2019). *The Value of Flexible Work: Evidence from Uber Drivers.* Journal of Political Economy 127(6):2735–2794. — Aggregate driver labor-supply elasticity ≈ 1.7; flexibility roughly doubles driver surplus.
- **[calib]** Angrist, J. D., Caldwell, S., & Hall, J. V. (2021). *Uber versus Taxi: A Driver's Eye View.* American Economic Journal: Applied Economics 13(3):272–308. — Intertemporal labor-supply elasticity ≈ 1.2–1.4 from a randomized lease experiment.
- **[calib]** Caldwell, S., & Oehlsen, E. (2022). *Gender, Outside Options, and Labor Supply: Experimental Evidence from the Gig Economy.* Working paper. — Heterogeneous by-type entry elasticities (≈0.3 for full-time men to ≈1.0 for part-time women); the supply-side source Castillo (2025) uses.
- **[calib/corpus]** Hall, J. V., Horton, J. J., & Knoepfle, D. T. (2023). *Ride-Sharing Markets Re-Equilibrate.* NBER Working Paper 30883. — After a fare increase, hourly earnings revert to baseline in roughly two months (business-stealing / entry on the hours margin).
- **[calib/corpus]** Cook, C., Diamond, R., Hall, J. V., List, J. A., & Oyer, P. (2021). *The Gender Earnings Gap in the Gig Economy: Evidence from over a Million Rideshare Drivers.* Review of Economic Studies 88(5):2210–2238. — ~7% gender earnings gap fully explained by driving speed (~48%), experience (~36%), and location choices (~28%); not customer discrimination.
- **[calib]** Goldszmidt, A., List, J. A., Metcalfe, R. D., Muir, I., Smith, V. K., & Wang, J. (2020). *The Value of Time in the United States: Estimates from Nationwide Natural Field Experiments.* NBER Working Paper 28208. — Value of time ≈ $19/hour (≈75% of after-tax wage) from Lyft field experiments.
- **[calib]** Abrantes, P. A. L., & Wardman, M. R. (2011). *Meta-analysis of UK values of travel time: An update.* Transportation Research Part A 45(1):1–17. — Waiting time valued ≈1.5× in-vehicle time.
- **[calib]** Larson, R. C., & Odoni, A. R. (1981). *Urban Operations Research.* Prentice-Hall. — Square-root law: mean distance to nearest point in a 2-D Poisson field ∝ (density)^(−1/2).
- **[calib]** Arnott, R. (1996). *Taxi Travel Should Be Subsidized.* Journal of Urban Economics 40(3):316–333. — Mean en-route (pickup) distance ∝ (idle-vehicle density)^(−1/2).
- **[calib/corpus]** Castillo, J. C., Knoepfle, D., & Weyl, E. G. (2017). *Surge Pricing Solves the Wild Goose Chase.* ACM EC 2017. — Pickup time T(I) ∝ I^(−1/n); 1/√I in 2-D.
- **[corpus]** Castillo, J. C. (2025). *Who Benefits from Surge Pricing?* Econometrica 93(5):1811–1854. — Structural model on Uber Houston data: surge raises total welfare +2.15%, rider surplus +3.57%, lowers driver surplus −0.98% and platform profit −0.50% (% of gross revenue); female and long-hours drivers hurt most. **Used only as a validation target, never as an input.**
- **[corpus]** Besbes, O., Castro, F., & Lobel, I. (2021). *Surge Pricing and Its Spatial Supply Response.* Management Science 67(3):1350–1367. — Local myopic surge is suboptimal; globally network-anticipating pricing dominates.
- **[corpus]** Afèche, P., Liu, Z., & Maglaras, C. (2023). *Ride-Hailing Networks with Strategic Drivers.* M&SOM 25(5):1890–1908. — Under strategic drivers, demand rejection can be optimal to induce repositioning.
- **[corpus]** Ma, H., Fang, F., & Parkes, D. C. (2019). *Spatio-Temporal Pricing for Ridesharing Platforms.* ACM EC 2019. — Welfare-optimal, envy-free spatiotemporal pricing mechanism.
- Schaller, B. (1999). *Elasticities for taxicab fares and service availability.* Transportation 26(3):283–297. — NYC taxi fare elasticity ≈ −0.22.
- Buchholz, N. (2022). *Spatial Equilibrium, Search Frictions, and Dynamic Efficiency in the Taxi Industry.* Review of Economic Studies 89(2):556–591.
- Frechette, G. R., Lizzeri, A., & Salz, T. (2019). *Frictions in a Competitive, Regulated Market: Evidence from Taxis.* American Economic Review 109(8):2954–2992.

## Learning / control for ride-hailing (corpus)

- **[corpus]** Xu, Z., Li, Z., Guan, Q., et al. (2018). *Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach.* KDD 2018. — Offline value learning + online Hungarian matching (the M3 template).
- **[corpus]** Lin, K., Zhao, R., Xu, Z., & Zhou, J. (2018). *Efficient Large-Scale Fleet Management via Multi-Agent Deep Reinforcement Learning.* KDD 2018.
- **[corpus]** Li, M., Qin, Z., Jiao, Y., et al. (2019). *Efficient Ridesharing Order Dispatching with Mean-Field Multi-Agent RL.* WWW 2019.
- **[corpus]** Eshkevari, S. S., Tang, X., Qin, Z., et al. (2022). *Reinforcement Learning in the Wild: Scalable RL Dispatching in a Ride-Hailing Marketplace.* KDD 2022.
- **[corpus]** Enders, T., Harrison, J., Pavone, M., & Schiffer, M. (2023). *Hybrid Multi-Agent Deep Reinforcement Learning for Autonomous Mobility on Demand.* L4DC 2023. — Per-agent actor + central weighted bipartite matching (the M2/hybrid template).
- **[corpus]** Gammelli, D., Yang, K., Harrison, J., et al. (2021). *Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems.* IEEE CDC 2021. — Bi-level GNN-RL + LP rebalancing (the R3 template).
- **[corpus]** Zhang, X., Varakantham, P., & Jiang, H. (2023). *Future-Aware Pricing and Matching for Sustainable On-Demand Ride Pooling.* AAAI 2023. — Joint future-aware pricing + matching.
- **[corpus]** Sun, J., Jin, H., Yang, Z., et al. (2024). *Optimizing Long-Term Efficiency and Fairness in Ride-Hailing via Joint Order Dispatching and Driver Repositioning (JDRCL).* IEEE TKDE 36(7):3348–3362. — Constrained max-min fairness MARL.
- **[corpus]** Yang, Z., Jin, H., et al. (2024). *Rethinking Order Dispatching in Online Ride-Hailing Platforms.* KDD 2024.
- **[corpus]** Han, et al. (2025). *GARLIC: GPT-Augmented Reinforcement Learning with Intelligent Control for Vehicle Dispatching.* AAAI 2025.
- **[corpus]** Lyu, et al. (2025). *LLM-ODDR: A Large Language Model Framework for Joint Order Dispatching and Driver Repositioning.* arXiv preprint.
