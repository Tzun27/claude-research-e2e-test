"""Results-chapter summary tables, saved to results/ as CSV + a text digest."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import selection as sel

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PRICE = {"P1": "flat", "P2": "reactive", "P3": "fluid"}
MATCH = {"M1": "greedy", "M2": "bipartite", "M3": "value"}
REBAL = {"R1": "none", "R2": "to-demand", "R3": "value-grad"}


def main():
    df = pd.read_csv(os.path.join(RES, "results.csv"))
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit(f"rows={len(df)}  scenarios={df.scenario.nunique()}  combos={df.combo.nunique()}  "
         f"seeds={df.seed.nunique()}")

    # 1. welfare decomposition by pricing family (mean over scenarios+seeds)
    emit("\n[T1] Welfare decomposition by pricing family (mean $):")
    t1 = df.groupby("price")[["rider_surplus", "ds_flexible", "ds_constrained",
                              "platform_profit", "total_welfare", "gmv",
                              "service_rate", "mean_surge"]].mean().round(1)
    emit(t1.to_string())
    t1.to_csv(os.path.join(RES, "t1_welfare_by_pricing.csv"))

    # 2. per-capita driver surplus by type & pricing (incidence)
    emit("\n[T2] Driver surplus per capita by type & pricing:")
    t2 = df.groupby("price")[["ds_flexible_per_capita", "ds_constrained_per_capita",
                              "earn_rate_flex", "earn_rate_cons"]].mean().round(2)
    t2["flex_minus_cons_pc"] = (t2.ds_flexible_per_capita - t2.ds_constrained_per_capita).round(2)
    emit(t2.to_string())
    t2.to_csv(os.path.join(RES, "t2_incidence_by_pricing.csv"))

    # 3. oracle frequency by objective
    emit("\n[T3] Oracle (best-combo) frequency by objective:")
    for kind in ["gmv", "welfare", "welfare_fair"]:
        piv = sel.aggregate(df, kind)
        oracle = piv.idxmax(axis=1)
        top = oracle.value_counts().head(6)
        emit(f"  {kind}: {oracle.nunique()} distinct oracle combos; top:")
        for combo, n in top.items():
            emit(f"      {combo:12s} {n:3d}/{len(oracle)}")

    # 4. GMV vs welfare divergence (RQ4)
    emit("\n[T4] Objective divergence (RQ4):")
    div = sel.oracle_divergence(df)
    n = len(div)
    emit(f"  GMV-oracle != welfare-oracle: {div['gmv_vs_welfare_differ'].sum()}/{n} scenarios")
    # also: pricing of gmv-oracle vs welfare-oracle
    gp = div["gmv"].str.split("+").str[1]; wp = div["welfare"].str.split("+").str[1]
    emit(f"  GMV-oracle pricing mix:     {gp.value_counts().to_dict()}")
    emit(f"  welfare-oracle pricing mix: {wp.value_counts().to_dict()}")
    # the category error: when we maximize GMV, what is the realized welfare regret?
    piv_w = sel.aggregate(df, "welfare"); _, reg_w = sel.oracle_and_regret(piv_w)
    piv_g = sel.aggregate(df, "gmv")
    gmv_oracle = piv_g.idxmax(axis=1)
    gmv_pick_welfare_regret = np.mean([reg_w.loc[s, gmv_oracle[s]] for s in reg_w.index])
    emit(f"  welfare-regret of deploying the GMV-optimal combo: {gmv_pick_welfare_regret:.3f}")

    # 5. selection regret summary (RQ3)
    emit("\n[T5] Selection regret (LOSO; lower=better):")
    _, summary = sel.run_all(os.path.join(RES, "results.csv"))
    rows = []
    for kind, s in summary.items():
        emit(f"  {kind:13s} oracle=0.000 random={s['random']:.3f} "
             f"single_fixed={s['single_best_fixed']:.3f} ({s['best_fixed_combo']}) "
             f"hand={s['hand_rule']:.3f} tree={s['selector_tree']:.3f} knn={s['selector_knn']:.3f}")
        rows.append({"objective": kind, **{k: s[k] for k in
                    ["random", "single_best_fixed", "hand_rule", "selector_tree",
                     "selector_knn", "best_fixed_combo", "n_distinct_oracle"]}})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "t5_selection_regret.csv"), index=False)

    with open(os.path.join(RES, "analysis_digest.txt"), "w") as f:
        f.write("\n".join(lines))
    emit("\nsaved tables + analysis_digest.txt to results/")


if __name__ == "__main__":
    main()
