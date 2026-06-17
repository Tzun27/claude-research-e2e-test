"""Generate all thesis figures from results.csv and the selection analysis.
Saves PNGs to thesis/figures/.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import selection as sel

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

PRICE_NAME = {"P1": "flat", "P2": "reactive surge", "P3": "fluid surge"}
MATCH_NAME = {"M1": "greedy", "M2": "bipartite", "M3": "value-based"}
REBAL_NAME = {"R1": "none", "R2": "to-demand", "R3": "value-grad"}


def load():
    return pd.read_csv(os.path.join(RES, "results.csv"))


def fig_welfare_decomposition(df):
    """Stacked welfare components averaged per pricing method (RQ1)."""
    g = df.groupby("price")[["rider_surplus", "ds_flexible", "ds_constrained",
                             "platform_profit"]].mean()
    g = g.loc[["P1", "P2", "P3"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(g))
    labels = {"rider_surplus": "rider surplus", "ds_flexible": "driver surplus (flexible)",
              "ds_constrained": "driver surplus (constrained)", "platform_profit": "platform profit"}
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    for c, col in zip(labels, colors):
        ax.bar([PRICE_NAME[i] for i in g.index], g[c], bottom=bottom, label=labels[c], color=col)
        bottom += g[c].values
    ax.set_ylabel("welfare ($, mean over scenarios & seeds)")
    ax.set_title("Welfare decomposition by pricing family")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_welfare_decomposition.png")); plt.close(fig)


def fig_gmv_vs_welfare(df):
    """GMV-maximizing vs welfare-maximizing diverge (the category error)."""
    piv_w = sel.aggregate(df, "welfare")
    piv_g = sel.aggregate(df, "gmv")
    # per scenario, normalize each combo's welfare and gmv to [0,1] then scatter the oracles
    fig, ax = plt.subplots(figsize=(6, 5.5))
    wsc = (piv_w.sub(piv_w.min(1), 0)).div((piv_w.max(1) - piv_w.min(1)), 0)
    gsc = (piv_g.sub(piv_g.min(1), 0)).div((piv_g.max(1) - piv_g.min(1)), 0)
    ax.scatter(gsc.values.ravel(), wsc.values.ravel(), s=8, alpha=0.25, color="#4C72B0",
               label="all combos")
    # highlight GMV-oracle and welfare-oracle per scenario
    for s in piv_w.index:
        go = piv_g.loc[s].idxmax(); wo = piv_w.loc[s].idxmax()
        ax.scatter(gsc.loc[s, go], wsc.loc[s, go], s=26, color="#C44E52", zorder=3)
        ax.scatter(gsc.loc[s, wo], wsc.loc[s, wo], s=26, marker="^", color="#55A868", zorder=3)
    ax.scatter([], [], color="#C44E52", label="GMV-optimal combo")
    ax.scatter([], [], marker="^", color="#55A868", label="welfare-optimal combo")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("normalized GMV (within scenario)")
    ax.set_ylabel("normalized total welfare (within scenario)")
    ax.set_title("GMV vs welfare across the method menu")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gmv_vs_welfare.png")); plt.close(fig)


def fig_incidence(df):
    """Per-capita driver surplus by type across pricing (RQ1 incidence)."""
    g = df.groupby("price")[["ds_flexible_per_capita", "ds_constrained_per_capita"]].mean()
    g = g.loc[["P1", "P2", "P3"]]
    x = np.arange(len(g)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w/2, g["ds_flexible_per_capita"], w, label="flexible", color="#55A868")
    ax.bar(x + w/2, g["ds_constrained_per_capita"], w, label="constrained", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels([PRICE_NAME[i] for i in g.index])
    ax.set_ylabel("driver surplus per capita ($)")
    ax.set_title("Surge incidence by driver type")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_incidence.png")); plt.close(fig)


def fig_regret(df):
    """Mean regret (+/- SEM across scenarios) per strategy & objective (RQ3)."""
    feats = sel.feature_frame(df)
    strategies = ["random", "hand_rule", "single_best_fixed",
                  "selector_tree_persub", "selector_tree_joint", "oracle"]
    labels = ["random", "hand-rule", "single-best-fixed", "selector\n(per-subtask)",
              "selector\n(joint)", "oracle"]
    objs = ["welfare", "gmv", "welfare_fair"]
    x = np.arange(len(strategies)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for j, o in enumerate(objs):
        piv = sel.aggregate(df, o); _, regret = sel.oracle_and_regret(piv)
        psr, _ = sel.per_scenario_regret(piv, regret, feats)
        means = [psr[s].mean() for s in strategies]
        sems = [psr[s].std(ddof=1) / np.sqrt(len(psr[s])) for s in strategies]
        ax.bar(x + (j-1)*w, means, w, yerr=sems, capsize=2, label=o.replace("_", "-"),
               color=colors[j], error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("normalized regret (lower is better)")
    ax.set_title("Context-to-method selection vs baselines (mean ± SEM over scenarios)")
    ax.legend(title="objective", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_regret.png")); plt.close(fig)


def fig_oracle_map(df):
    """Which pricing family is welfare-optimal across (dsr, concentration)."""
    piv = sel.aggregate(df, "welfare")
    oracle = piv.idxmax(axis=1)
    feats = sel.feature_frame(df)
    price = oracle.str.split("+").str[1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = {"P1": "#4C72B0", "P2": "#DD8452", "P3": "#C44E52"}
    for p in ["P1", "P2", "P3"]:
        m = price == p
        ax.scatter(feats.loc[m, "demand_supply_ratio"], feats.loc[m, "spatial_concentration"],
                   c=cmap[p], label=f"{p} ({PRICE_NAME[p]})", s=60, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("demand/supply ratio"); ax.set_ylabel("spatial concentration")
    ax.set_title("Welfare-optimal pricing family by context")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_oracle_map.png")); plt.close(fig)


def fig_castillo_regime():
    """Incidence by market regime from the saved Castillo sweep (validation)."""
    path = os.path.join(RES, "castillo_regime_sweep.csv")
    if not os.path.exists(path):
        return
    s = pd.read_csv(path)
    s["label"] = s.apply(lambda r: f"dsr={r.dsr}\nconc={r.concentration}", axis=1)
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.bar(x - w/2, s["d_ds_flexible"], w, label="Δ flexible driver surplus", color="#55A868")
    ax.bar(x + w/2, s["d_ds_constrained"], w, label="Δ constrained driver surplus", color="#C44E52")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(s["label"], fontsize=7)
    ax.set_ylabel("Δ driver surplus, surge − welfare-optimal uniform ($)")
    ax.set_title("Smart surge is regressive across drivers: flexible always gain more than\n"
                 "constrained; constrained are absolutely hurt only in mild scarcity (dsr=0.8)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_castillo_regime.png")); plt.close(fig)


def fig_selection_box(df):
    """Per-scenario regret distribution (welfare) and the paired selector-vs-fixed
    difference, showing the selector's mean edge is driven by a few scenarios."""
    feats = sel.feature_frame(df)
    piv = sel.aggregate(df, "welfare")
    _, regret = sel.oracle_and_regret(piv)
    psr, bf = sel.per_scenario_regret(piv, regret, feats)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    order = ["random", "hand_rule", "single_best_fixed", "selector_tree_joint"]
    labs = ["random", "hand-rule", "single-fixed", "selector\n(joint)"]
    data = [psr[k] for k in order]
    bp = ax1.boxplot(data, tick_labels=labs, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#937860", "#DD8452", "#55A868"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax1.set_ylabel("per-scenario normalized regret (welfare)")
    ax1.set_title("Regret distribution across scenarios (LOSO)")
    # paired difference: fixed - selector (positive => selector better)
    d = np.sort(psr["single_best_fixed"] - psr["selector_tree_joint"])
    ax2.bar(np.arange(len(d)), d, color=np.where(d >= 0, "#55A868", "#C44E52"))
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("scenario (sorted)"); ax2.set_ylabel("regret(fixed) − regret(selector)")
    ax2.set_title("Paired gain of selector over single-fixed\n(>0: selector better; the edge is a few outliers)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_selection_box.png")); plt.close(fig)


def fig_validation():
    """Regenerate V1 (elasticity) and V2 (square-root law) data and plot."""
    import validate as val
    from scenarios import elas_to_sigma
    from config import ScenarioConfig
    base = ScenarioConfig(name="calib", grid_n=5, n_steps=140, demand_supply_ratio=0.9,
                          fleet_size=120, spatial_concentration=0.5, temporal_peakedness=2.5,
                          demand_elasticity=-0.55, flex_frac=0.5, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # V1: elasticity vs sigma
    sig_grid = np.linspace(0.5, 1.4, 8)
    es = [val.measure_elasticity(base, s, n_seeds=4) for s in sig_grid]
    axes[0].plot(sig_grid, es, "o-", color="#4C72B0")
    axes[0].axhline(-0.55, color="#C44E52", ls="--", label="target −0.55 (Cohen 2016)")
    axes[0].set_xlabel("value-spread σ_v"); axes[0].set_ylabel("realized demand elasticity")
    axes[0].set_title("V1: demand elasticity calibration"); axes[0].legend(fontsize=8)
    # V2: square-root law scatter + fit
    from simulator import Market
    from methods import make_methods
    from dataclasses import replace
    dens, dist = [], []
    sig = elas_to_sigma(-0.55)
    for fleet in (70, 100, 130, 170, 210):
        sc = replace(base, seed=7, fleet_size=fleet, demand_supply_ratio=0.8)
        m, p, r = make_methods(("M2", "P1", "R1"))
        mkt = Market(sc, p, m, r, sigma_v=sig, collect_traces=True); mkt.run()
        dens.extend(mkt.trace_local_density); dist.extend(mkt.trace_pickup_dist)
    dens = np.array(dens, float); dist = np.array(dist, float)
    mask = dens > 0.2; dens, dist = dens[mask], dist[mask]
    qs = np.quantile(dens, np.linspace(0, 1, 13)); xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        b = (dens >= lo) & (dens <= hi)
        if b.sum() >= 20: xs.append(dens[b].mean()); ys.append(dist[b].mean())
    xs, ys = np.array(xs), np.array(ys)
    a, corr, _ = val.sqrt_law(base, sig)
    axes[1].loglog(xs, ys, "o", color="#55A868", label="binned means")
    k = np.exp(np.mean(np.log(ys) + a * np.log(xs)))
    axes[1].loglog(xs, k * xs ** (-a), "-", color="#C44E52",
                   label=f"fit ∝ ρ^(−{a:.2f})")
    axes[1].set_xlabel("local idle density ρ"); axes[1].set_ylabel("mean pickup distance (cells)")
    axes[1].set_title(f"V2: square-root law (a={a:.2f}, |corr|={abs(corr):.2f})")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_validation.png")); plt.close(fig)


def main():
    df = load()
    fig_welfare_decomposition(df)
    fig_gmv_vs_welfare(df)
    fig_incidence(df)
    fig_regret(df)
    fig_oracle_map(df)
    fig_castillo_regime()
    fig_selection_box(df)
    fig_validation()
    print("wrote figures to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
