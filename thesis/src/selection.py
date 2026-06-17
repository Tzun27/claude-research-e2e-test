"""Oracle, context->method selector, and regret evaluation (DESIGN RQ3, RQ4).

The brute-force oracle (best combo per scenario, computed from the full results
matrix) is the gold standard -- NOT a competitor. The question is whether a cheap
selector that sees ONLY observable city context can pick a near-oracle combo. We
evaluate by normalized regret under leave-one-scenario-out (LOSO) cross-validation
and compare against: random, single-best-fixed ("use one method everywhere"),
and a hand-written rule.

We also ask (RQ4) whether the oracle combo itself changes with the objective
(GMV vs welfare vs fairness-adjusted welfare) -- if it does, optimizing GMV
systematically mis-selects from a welfare standpoint.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FEATURES = ["demand_supply_ratio", "spatial_concentration", "temporal_peakedness",
            "demand_elasticity", "flex_frac", "trip_length_mean", "grid_n"]
LAM_FAIR = 1.0


def objective(df, kind):
    """Return a Series of objective values per row."""
    if kind == "welfare":
        return df["total_welfare"]
    if kind == "gmv":
        return df["gmv"]
    if kind == "welfare_fair":
        gap = np.maximum(0.0, df["ds_flexible_per_capita"] - df["ds_constrained_per_capita"])
        gap_dollars = gap * df["n_constrained"]
        return df["total_welfare"] - LAM_FAIR * gap_dollars
    raise ValueError(kind)


def aggregate(df, kind):
    """Mean objective per (scenario, combo). Returns pivot scenario x combo."""
    d = df.copy()
    d["obj"] = objective(d, kind)
    g = d.groupby(["scenario", "combo"])["obj"].mean().reset_index()
    pivot = g.pivot(index="scenario", columns="combo", values="obj")
    return pivot


def oracle_and_regret(pivot):
    """Per scenario: oracle combo, and normalized regret of every combo in [0,1]."""
    best = pivot.max(axis=1)
    worst = pivot.min(axis=1)
    rng = (best - worst).replace(0, np.nan)
    regret = (best.values[:, None] - pivot.values) / rng.values[:, None]
    regret = pd.DataFrame(regret, index=pivot.index, columns=pivot.columns).fillna(0.0)
    oracle = pivot.idxmax(axis=1)
    return oracle, regret


def feature_frame(df):
    """One row of context features per scenario."""
    f = df.groupby("scenario")[FEATURES].first()
    return f


def loso_selector_regret(pivot, regret, feats, model="tree", per_subtask=True):
    """Leave-one-scenario-out regret of a context->method selector."""
    scen = list(pivot.index)
    X = feats.loc[scen].values
    combos = list(pivot.columns)
    oracle = pivot.idxmax(axis=1)
    regrets = []
    chosen = {}
    for i, s in enumerate(scen):
        tr = [j for j in range(len(scen)) if j != i]
        Xtr, Xte = X[tr], X[i:i+1]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        if per_subtask:
            # predict each sub-task independently, then compose
            picks = {}
            for pos, sub in [(0, "match"), (1, "price"), (2, "rebal")]:
                ylab = np.array([oracle.iloc[j].split("+")[pos] for j in tr])
                clf = _mk(model)
                if len(set(ylab)) == 1:
                    picks[sub] = ylab[0]
                else:
                    clf.fit(Xtr_s, ylab)
                    picks[sub] = clf.predict(Xte_s)[0]
            combo = f"{picks['match']}+{picks['price']}+{picks['rebal']}"
        else:
            ylab = np.array([oracle.iloc[j] for j in tr])
            clf = _mk(model)
            clf.fit(Xtr_s, ylab)
            combo = clf.predict(Xte_s)[0]
        if combo not in regret.columns:        # composed combo always valid (27 exist)
            combo = oracle.iloc[tr[0]]
        regrets.append(regret.loc[s, combo])
        chosen[s] = combo
    return float(np.mean(regrets)), regrets, chosen


def _mk(model):
    if model == "tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=0)
    if model == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    raise ValueError(model)


def baseline_regrets(pivot, regret):
    """random, single-best-fixed, oracle (=0)."""
    combos = list(pivot.columns)
    # random: mean regret across all combos per scenario
    rnd = regret.mean(axis=1).mean()
    # single best fixed: combo with best mean objective across scenarios
    best_fixed = pivot.mean(axis=0).idxmax()
    bf = regret[best_fixed].mean()
    return {"oracle": 0.0, "random": float(rnd),
            "single_best_fixed": float(bf), "_best_fixed_combo": best_fixed}


def hand_rule(feats):
    """A simple domain decision table: context -> combo (the human baseline).
    Pricing scales with scarcity: flat when slack, reactive when moderate, fluid
    clearing when scarce (all three families are reachable)."""
    out = {}
    for s, row in feats.iterrows():
        dsr = row["demand_supply_ratio"]; conc = row["spatial_concentration"]
        peak = row["temporal_peakedness"]
        match = "M3" if (dsr > 1.0 or conc > 0.6) else "M2"
        if dsr > 0.9:
            price = "P3"                  # scarce: clear the market
        elif dsr < 0.65 and peak < 2.0:
            price = "P1"                  # slack & flat demand: no surge
        else:
            price = "P2"                  # in between: mild reactive surge
        rebal = "R2" if conc > 0.5 else "R1"
        out[s] = f"{match}+{price}+{rebal}"
    return out


def hand_rule_regret(regret, feats):
    picks = hand_rule(feats)
    return float(np.mean([regret.loc[s, picks[s]] for s in regret.index])), picks


def oracle_divergence(df):
    """RQ4: do oracle combos differ across objectives? Return per-scenario table."""
    objs = ["gmv", "welfare", "welfare_fair"]
    cols = {}
    for o in objs:
        piv = aggregate(df, o)
        cols[o] = piv.idxmax(axis=1)
    tab = pd.DataFrame(cols)
    tab["gmv_vs_welfare_differ"] = tab["gmv"] != tab["welfare"]
    return tab


def selector_rules(df, kind="welfare"):
    """Interpretability: per-subtask feature importances + the pricing decision
    tree (text). Fit on all scenarios (not LOSO) for inspection."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    feats = feature_frame(df)
    piv = aggregate(df, kind)
    oracle = piv.idxmax(axis=1)
    X, cols = feats.values, list(feats.columns)
    out = {"importances": {}, "oracle_subtask_dist": {}}
    for pos, sub in [(0, "match"), (1, "price"), (2, "rebal")]:
        y = oracle.str.split("+").str[pos].values
        out["oracle_subtask_dist"][sub] = dict(pd.Series(y).value_counts())
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=0).fit(X, y)
        out["importances"][sub] = {c: round(float(v), 3)
                                   for c, v in zip(cols, clf.feature_importances_) if v > 0.01}
    yp = oracle.str.split("+").str[1].values
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(X, yp)
    out["pricing_tree"] = export_text(clf, feature_names=cols, max_depth=3)
    return out


def per_scenario_regret(pivot, regret, feats):
    """Per-scenario regret arrays (indexed by scenario) for every strategy."""
    scen = list(pivot.index)
    bf = pivot.mean(axis=0).idxmax()
    out = {
        "oracle": np.zeros(len(scen)),
        "random": regret.mean(axis=1).loc[scen].values,
        "single_best_fixed": regret[bf].loc[scen].values,
    }
    hr, picks = hand_rule_regret(regret, feats)
    out["hand_rule"] = np.array([regret.loc[s, picks[s]] for s in scen])
    for model in ("tree", "knn"):
        for ps, tag in ((True, "_persub"), (False, "_joint")):
            _, _, pk = loso_selector_regret(pivot, regret, feats, model, per_subtask=ps)
            out[f"selector_{model}{tag}"] = np.array([regret.loc[s, pk[s]] for s in scen])
    return out, bf


def paired_vs_fixed(strategy_regret, fixed_regret, rng=None):
    """Paired comparison (fixed - strategy); positive => strategy better."""
    rng = rng or np.random.default_rng(0)
    d = fixed_regret - strategy_regret
    boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-9]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    return {"mean_gain": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "wilcoxon_p": p, "win": int((d > 1e-9).sum()),
            "lose": int((d < -1e-9).sum()), "tie": int((np.abs(d) <= 1e-9).sum())}


def run_all(results_csv=None):
    results_csv = results_csv or os.path.join(RESULTS_DIR, "results.csv")
    df = pd.read_csv(results_csv)
    feats = feature_frame(df)
    summary = {}
    for kind in ["welfare", "gmv", "welfare_fair"]:
        piv = aggregate(df, kind)
        oracle, regret = oracle_and_regret(piv)
        base = baseline_regrets(piv, regret)
        tree_r, _, tree_pick = loso_selector_regret(piv, regret, feats, "tree")
        knn_r, _, _ = loso_selector_regret(piv, regret, feats, "knn")
        hr_r, _ = hand_rule_regret(regret, feats)
        summary[kind] = {
            "oracle": base["oracle"], "random": base["random"],
            "single_best_fixed": base["single_best_fixed"],
            "best_fixed_combo": base["_best_fixed_combo"],
            "hand_rule": hr_r, "selector_tree": tree_r, "selector_knn": knn_r,
            "n_distinct_oracle": oracle.nunique(),
        }
    return df, summary


if __name__ == "__main__":
    df, summary = run_all()
    print(f"loaded {len(df)} rows, {df['scenario'].nunique()} scenarios, "
          f"{df['combo'].nunique()} combos\n")
    print("=== Normalized regret (lower=better; oracle=0, random~0.5) ===")
    hdr = ["objective", "oracle", "random", "single_fixed", "hand_rule", "sel_tree", "sel_knn", "#oracle"]
    print("  " + "  ".join(f"{h:>12s}" for h in hdr))
    for kind, s in summary.items():
        print("  " + "  ".join(f"{v:>12}" if isinstance(v, str) else f"{v:>12.3f}" for v in
              [kind, s["oracle"], s["random"], s["single_best_fixed"], s["hand_rule"],
               s["selector_tree"], s["selector_knn"]]) + f"  {s['n_distinct_oracle']:>6d}")
    print("\nsingle-best-fixed combo per objective:")
    for kind, s in summary.items():
        print(f"  {kind:14s} {s['best_fixed_combo']}")
    print("\n=== RQ4: oracle divergence across objectives ===")
    tab = oracle_divergence(df)
    print(f"  scenarios where GMV-oracle != welfare-oracle: "
          f"{tab['gmv_vs_welfare_differ'].sum()}/{len(tab)}")
