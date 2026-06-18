"""Aggregate result JSONs into tables and figures for the thesis.

Reads results/data/*.json (produced by run_cell.py) and emits:
  - results/tables/*.md   : four-way welfare-incidence tables (Delta % of gross revenue),
                            driver-type incidence, fairness frontier, sensitivity.
  - results/figures/*.png : welfare incidence bars, incidence by driver type,
                            efficiency-equity frontier, sensitivity curves.
All deltas are surge - uniform, normalized by uniform gross revenue, with bootstrap CIs
over evaluation (market) seeds.
"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA = os.path.join(os.path.dirname(__file__), "..", "results", "data")
TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
FIG = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

CASTILLO = dict(total_welfare=2.15, rider_surplus=3.57, driver_surplus=-0.98, platform_profit=-0.50)
COMPONENTS = ["total_welfare", "rider_surplus", "driver_surplus", "platform_profit"]


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, pattern))):
        with open(f) as fh:
            out[os.path.basename(f)[:-5]] = json.load(fh)
    return out


def paired_delta_pctGR(res, baseline_key="uniform_summaries"):
    """Return dict component -> (mean %GR, sem %GR) for surge-baseline, paired by eval seed.

    baseline_key: 'uniform_summaries' (optimal uniform) or 'matched_uniform_summaries'
    (uniform at the controller's own average multiplier -> the mean-preserving-spread test).
    surge_summaries may pool multiple train seeds; we average surge over train-seed replicas
    per eval seed, then pair against the baseline per eval seed.
    """
    uni = res[baseline_key]; sur = res["surge_summaries"]
    eval_seeds = res["eval_seeds"]; nE = len(eval_seeds)
    GR = np.mean([u["gross_revenue"] for u in uni])
    # group surge by eval-seed index (pooled in order train-major: [seed-block per train seed])
    n_train = len(sur) // nE
    out = {}
    for comp in COMPONENTS:
        u = np.array([uni[i][comp] for i in range(nE)])
        s = np.array([[sur[t * nE + i][comp] for i in range(nE)] for t in range(n_train)]).mean(0)
        d = (s - u) / GR * 100.0
        out[comp] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(nE) if nE > 1 else 0.0))
    out["_GR"] = GR
    out["_uniform_mult"] = res["uniform_mult"]
    return out


def fmt_signs(res_delta):
    """Compare signs to Castillo; return a match string."""
    match = []
    for comp in COMPONENTS:
        s = np.sign(res_delta[comp][0]); cs = np.sign(CASTILLO[comp])
        match.append("OK" if s == cs else "x")
    return match


def table_welfare_incidence(cells, title, fname):
    lines = [f"# {title}", "",
             "Delta = surge - uniform, as % of uniform gross revenue (mean +/- s.e.m. over market seeds).",
             "Castillo (2025) targets: total +2.15, rider +3.57, driver -0.98, platform -0.50 (% of GR).", ""]
    header = "| objective | uniform m* | dW | d rider | d driver | d platform | signs vs Castillo |"
    lines += [header, "|" + "---|" * 7]
    for name, res in cells.items():
        d = paired_delta_pctGR(res)
        sg = fmt_signs(d)
        obj = res["objective"]
        def cell(c, i):
            return f"{d[c][0]:+.2f}±{d[c][1]:.2f}"
        signstr = " ".join(f"{c.split('_')[0]}:{m}" for c, m in zip(COMPONENTS, sg))
        lines.append(f"| {obj} | {d['_uniform_mult']:.2f} | {cell('total_welfare',0)} | "
                     f"{cell('rider_surplus',1)} | {cell('driver_surplus',2)} | "
                     f"{cell('platform_profit',3)} | {signstr} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, fname), "w") as f:
        f.write(txt)
    print(txt)
    return txt


def table_mps(cells, fname):
    """RQ1 (calibration-robust): surge vs uniform AT THE SAME AVERAGE multiplier
    (Castillo's mean-preserving-spread comparison). Isolates the effect of price *variation*."""
    lines = ["# RQ1: welfare incidence of price *variation* (mean-preserving spread)", "",
             "Delta = surge - uniform-at-the-same-average-multiplier, % of GR (mean +/- s.e.m.).",
             "This isolates the pure effect of spatio-temporal price *variation* (allocative",
             "efficiency + matching), controlling for the price level. Castillo signs: +,+,-,-.", "",
             "| objective | surge avg mult | dW | d rider | d driver | d platform | signs |",
             "|" + "---|" * 7]
    for name, res in cells.items():
        if "matched_uniform_summaries" not in res:
            continue
        d = paired_delta_pctGR(res, "matched_uniform_summaries")
        sg = fmt_signs(d)
        signstr = " ".join(f"{c.split('_')[0]}:{m}" for c, m in zip(COMPONENTS, sg))
        lines.append(f"| {res['objective']} | {res.get('surge_mean_mult',0):.2f} | "
                     f"{d['total_welfare'][0]:+.2f}±{d['total_welfare'][1]:.2f} | "
                     f"{d['rider_surplus'][0]:+.2f} | {d['driver_surplus'][0]:+.2f} | "
                     f"{d['platform_profit'][0]:+.2f} | {signstr} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, fname), "w") as f:
        f.write(txt)
    print(txt)
    return txt


def fig_welfare_incidence(cells, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    objs = [res["objective"] for res in cells.values()]
    deltas = [paired_delta_pctGR(res) for res in cells.values()]
    x = np.arange(len(COMPONENTS)); w = 0.8 / max(1, len(objs))
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (obj, d) in enumerate(zip(objs, deltas)):
        vals = [d[c][0] for c in COMPONENTS]; errs = [d[c][1] for c in COMPONENTS]
        ax.bar(x + j * w, vals, w, yerr=errs, capsize=3, label=obj)
    # Castillo reference markers
    cv = [CASTILLO[c] for c in COMPONENTS]
    ax.plot(x + 0.4 - w/2, cv, "k*", markersize=14, label="Castillo 2025")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + 0.4 - w/2); ax.set_xticklabels([c.replace("_", "\n") for c in COMPONENTS])
    ax.set_ylabel("Δ (surge − uniform), % of gross revenue")
    ax.set_title("Welfare incidence of learned surge pricing, by platform objective")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=140); plt.close(fig)
    print(f"saved {fname}")


def table_rq2_levels(cells, fname):
    """RQ2 (central): how each objective's learned surge SPLITS welfare among the parties.

    Reports the DISTRIBUTION (rider/driver/platform as % of total welfare) -- comparable
    across objectives, unlike % of gross revenue (whose denominator moves with the price
    level) -- plus the absolute total welfare (normalized to the max across objectives) so
    efficiency is visible too."""
    Wmax = max(res["surge_agg"]["total_welfare"]["mean"] for res in cells.values())
    lines = ["# RQ2: welfare incidence by platform objective (learned surge)", "",
             "Shares are % of TOTAL WELFARE (the distribution among parties); total welfare is",
             "indexed to the max-welfare objective (=100). The same surge mechanism splits",
             "welfare very differently depending on the objective.", "",
             "| objective | surge avg mult | rider %W | driver %W | platform %W | total welfare (idx) |",
             "|" + "---|" * 6]
    for name, res in cells.items():
        s = res["surge_agg"]; W = s["total_welfare"]["mean"]
        lines.append(f"| {res['objective']} | {res.get('surge_mean_mult',0):.2f} | "
                     f"{100*s['rider_surplus']['mean']/W:.0f} | {100*s['driver_surplus']['mean']/W:.0f} | "
                     f"{100*s['platform_profit']['mean']/W:.0f} | {100*W/Wmax:.0f} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, fname), "w") as f:
        f.write(txt)
    print(txt)
    return txt


def table_threeway(price_cells, tw_cells, fname):
    """Compare price-only vs three-way control welfare incidence for shared objectives."""
    lines = ["# Three-way control vs price-only (Gap A0)", "",
             "Δ (surge − uniform) % of GR. Does adding matching+rebalancing control shift the incidence?", ""]
    lines += ["| objective | condition | dW | d rider | d driver | d platform |", "|" + "---|" * 6]
    for cells, label in [(price_cells, "price"), (tw_cells, "three-way")]:
        for name, res in cells.items():
            d = paired_delta_pctGR(res)
            lines.append(f"| {res['objective']} | {label} | "
                         f"{d['total_welfare'][0]:+.2f} | {d['rider_surplus'][0]:+.2f} | "
                         f"{d['driver_surplus'][0]:+.2f} | {d['platform_profit'][0]:+.2f} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, fname), "w") as f:
        f.write(txt)
    print(txt)


def fig_fairness_frontier(fair_cells, base_cell, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    cells = dict(base_cell); cells.update(fair_cells)
    for name, res in cells.items():
        agg = res["surge_agg"]
        # equity = -(across-type per-capita surplus dispersion); efficiency = total welfare
        bt = agg["driver_by_type"]
        pc = [bt[t]["surplus"]["mean"] / max(1, bt[t]["n_drivers"]["mean"]) for t in bt]
        disp = max(pc) - min(pc)
        rows.append((res.get("fair_weight", 0.0), agg["total_welfare"]["mean"],
                     agg["driver_surplus"]["mean"], disp))
    rows.sort()
    w, W, DS, disp = zip(*rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(disp, W, "o-")
    for wi, di, Wi in zip(w, disp, W):
        ax[0].annotate(f"λ={wi:g}", (di, Wi), fontsize=8)
    ax[0].set_xlabel("driver-earnings dispersion across types (inequity)")
    ax[0].set_ylabel("total welfare (efficiency)")
    ax[0].set_title("Efficiency–equity frontier")
    ax[1].plot(w, DS, "s-", label="driver surplus")
    ax[1].plot(w, disp, "^-", label="type dispersion")
    ax[1].set_xlabel("fairness weight λ"); ax[1].legend(); ax[1].set_title("Effect of fairness penalty")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=140); plt.close(fig)
    print(f"saved {fname}")


def table_sensitivity(sens_cells, base_cell, fname):
    lines = ["# Scope conditions: welfare incidence across market regimes (welfare-weighted, price)", "",
             "Δ (surge − uniform) % of GR. Tests when the Castillo sign structure holds.", "",
             "| variant | uniform m* | dW | d rider | d driver | d platform | structure |",
             "|" + "---|" * 7]
    cells = dict(base_cell); cells.update(sens_cells)
    for name, res in cells.items():
        d = paired_delta_pctGR(res)
        sg = fmt_signs(d)
        struct = "Castillo" if sg == ["OK", "OK", "OK", "OK"] else "/".join(sg)
        var = name.replace("sens_", "").replace("price_welfare_weighted", "base")
        lines.append(f"| {var} | {d['_uniform_mult']:.2f} | {d['total_welfare'][0]:+.2f} | "
                     f"{d['rider_surplus'][0]:+.2f} | {d['driver_surplus'][0]:+.2f} | "
                     f"{d['platform_profit'][0]:+.2f} | {struct} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, fname), "w") as f:
        f.write(txt)
    print(txt)


def table_driver_incidence(cells, fname):
    """By-driver-type surplus change (surge - uniform), per capita (RQ3)."""
    lines = ["# Driver-incidence by type (RQ3): per-capita surplus change, surge − uniform", "",
             "Tests whether the most-engaged (full-time, low reservation-wage) drivers lose most.", ""]
    for name, res in cells.items():
        u = res["uniform_agg"]["driver_by_type"]; s = res["surge_agg"]["driver_by_type"]
        lines.append(f"\n**{res['objective']} ({res['condition']})**\n")
        lines += ["| type | uniform $/driver | surge $/driver | Δ $/driver |", "|" + "---|" * 4]
        for t in u:
            un = u[t]["surplus"]["mean"] / max(1, u[t]["n_drivers"]["mean"])
            sn = s[t]["surplus"]["mean"] / max(1, s[t]["n_drivers"]["mean"])
            lines.append(f"| {t} | {un:.2f} | {sn:.2f} | {sn-un:+.2f} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, fname), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    price = load("price_*.json")
    if price:
        order = ["profit", "throughput", "welfare_weighted", "welfare"]
        price = {k: v for o in order for k, v in price.items() if v["objective"] == o}
        table_mps(price, "rq1_mean_preserving_spread.md")
        table_rq2_levels(price, "rq2_incidence_by_objective.md")
        table_welfare_incidence(price, "Welfare incidence vs OPTIMAL uniform (price level + variation)",
                                "welfare_incidence_price.md")
        fig_welfare_incidence(price, "welfare_incidence_price.png")
        table_driver_incidence(price, "driver_incidence.md")
    else:
        print("No price_*.json results yet.")

    tw = load("threeway_*.json")
    if tw and price:
        shared = {k: v for k, v in price.items() if v["objective"] in [r["objective"] for r in tw.values()]}
        table_threeway(shared, tw, "threeway_vs_price.md")

    fair = load("fair_*.json")
    base_ww = {k: v for k, v in price.items() if v["objective"] == "welfare_weighted"} if price else {}
    if fair and base_ww:
        fig_fairness_frontier(fair, base_ww, "fairness_frontier.png")

    sens = load("sens_*.json")
    if sens and base_ww:
        table_sensitivity(sens, base_ww, "sensitivity.md")
