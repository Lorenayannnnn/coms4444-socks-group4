"""Reproduce report figures from results/7_raw_runs.csv using plain Matplotlib.

Run with a Python environment containing numpy, pandas and matplotlib.
No simulator runs, fitted models, or external data are used.
"""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/socks-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import FuncFormatter

# One accent hue for Group 4, a second hue only for the ablation, gray for the
# rest of the field, and a single-hue blue ramp for magnitudes.
BLUE, ORANGE, GRAY, RULE = "#2a78d6", "#eb6834", "#a8a7a2", "#e4e3df"
INK, INK2, MUTED, BAND = "#0b0b0b", "#52514e", "#8a8984", "#f3f2ee"
RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": .6, "axes.axisbelow": True,
    "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 6,
    "legend.frameon": False, "legend.fontsize": 9.5, "pdf.fonttype": 42,
})
COMMA = FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", "\u2212"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def heatmap(ax, data, title, fmt, vmax=None, gamma=1.0):
    vals = data.to_numpy(dtype=float)
    vmax = vmax if vmax is not None else np.nanmax(vals)
    norm = PowerNorm(gamma, vmin=0, vmax=vmax)
    ax.imshow(vals, aspect="auto", cmap=RAMP, norm=norm)
    ax.set_xticks(range(len(data.columns)), data.columns)
    ax.set_yticks(range(len(data.index)), data.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(data.columns)), minor=True)
    ax.set_yticks(np.arange(-.5, len(data.index)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            dark = norm(v) > .55
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8.6,
                    color="white" if dark else INK)


x = pd.read_csv(ROOT / "results/7_raw_runs.csv", low_memory=False)
assert not x.duplicated(["sim_id", "seat"]).any()
assert len(x) == 578016 and x.sim_id.nunique() == 81216
assert (x.groupby("sim_id").size() == x.groupby("sim_id").roommates.first()).all()
config_keys = ["household_type", "roster", "unit", "drawer_multiplier", "budget_total", "days"]
assert x.drop_duplicates("sim_id").groupby(config_keys, dropna=False).size().eq(3).all()
x["mismatch_total"] = x.total_embarrassment - 65536 * x.sockless_days
assert (x.mismatch_total >= 0).all()
x["mismatch_daily"] = x.mismatch_total / x.days
x["sockless_fraction"] = x.sockless_days / x.days
x["dressed_days"] = x.days - x.sockless_days
x["had_sockless"] = x.sockless_days.gt(0)

meta = ["household_type", "household_description", "roommates", "roster", "unit",
        "drawer_socks", "drawer_multiplier", "budget_total", "years", "days", "seed",
        "household_total_spent", "household_spend_per_year", "budget_exhausted_on_day"]
assert x.groupby("sim_id")[meta].nunique(dropna=False).max().max() == 1
h = x.groupby("sim_id", as_index=False).agg(
    **{c: (c, "first") for c in meta},
    score=("embarrassment_per_day", "mean"),
    mismatch_daily=("mismatch_daily", "mean"),
    sockless_fraction=("sockless_fraction", "mean"),
    any_sockless=("had_sockless", "max"),
)
h["exhausted"] = h.budget_exhausted_on_day.notna()
h["exhaustion_fraction"] = h.budget_exhausted_on_day / h.days
h["rho"] = 10 * h.roommates * h.days / (3 * h.budget_total)
h["budget_per_roommate_year"] = h.budget_total / (h.roommates * h.years)
h["roster_tuple"] = h.roster.map(lambda s: tuple(sorted(map(int, s.split("+")))))

g = x.groupby(["sim_id", "group"], as_index=False).agg(
    score=("embarrassment_per_day", "mean"),
    total=("total_embarrassment", "mean"),
    mismatch_daily=("mismatch_daily", "mean"),
    mismatch_total=("mismatch_total", "mean"),
    sockless_days=("sockless_days", "mean"),
    sockless_fraction=("sockless_fraction", "mean"),
    dressed_days=("dressed_days", "mean"),
    any_sockless=("had_sockless", "max"),
    role=("role", "first"),
)
g = g.merge(h[["sim_id"] + meta + ["exhausted", "exhaustion_fraction", "rho",
                                         "budget_per_roommate_year", "roster_tuple"]],
            on="sim_id", validate="many_to_one")
g4 = g[g.group == 4].copy()
homo = g[g.household_type.isin(["n1", "n2_homo", "n9_homo", "n18_homo", "n36_homo"])]
g4h = homo[homo.group == 4]
assert len(g4) == 29376

# Figure 1: equal weight to a simulation containing a group, not to its seats.
overall = g.groupby("group").agg(
    runs=("sim_id", "size"), mean=("score", "mean"), median=("score", "median"),
    mismatch_daily=("mismatch_daily", "mean"),
    sockless_run_pct=("any_sockless", lambda s: 100*s.mean()),
    sockless_day_pct=("sockless_fraction", lambda s: 100*s.mean()),
    exhaustion_pct=("exhausted", lambda s: 100*s.mean()),
)
overall["mismatch_per_dressed_day"] = g.groupby("group").mismatch_total.sum() / g.groupby("group").dressed_days.sum()
overall["seat_weighted_mean"] = x.groupby("group").embarrassment_per_day.mean()
overall["rank_simulation"] = overall["mean"].rank(method="min").astype(int)
overall["rank_seat"] = overall.seat_weighted_mean.rank(method="min").astype(int)
overall.to_csv(OUT / "overall.csv")
supplied = pd.read_csv(ROOT / "results/1_overall_standings.csv").set_index("group")
assert np.allclose(overall.seat_weighted_mean, supplied.embarrassment_per_day_mean.reindex(overall.index), atol=.005)
seed_summary = g.groupby(["group", "seed"]).agg(mean=("score", "mean"),
    mismatch_total=("mismatch_total", "sum"), dressed_days=("dressed_days", "sum"))
seed_summary["mismatch_per_dressed_day"] = seed_summary.mismatch_total / seed_summary.dressed_days
seed_summary.to_csv(OUT / "seed_summary.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"wspace": .75})
for ax, col, title, xlabel, fmt in [
        (axs[0], "mean", "Total embarrassment", "Per roommate-day", "{:,.1f}"),
        (axs[1], "mismatch_per_dressed_day", "Mismatch embarrassment", "Per dressed roommate-day", "{:.3f}")]:
    order = overall[col].sort_values(ascending=False).index
    repeated = seed_summary[col].unstack().reindex(order)
    for i, grp in enumerate(order):
        c = BLUE if grp == 4 else GRAY
        ax.plot([repeated.loc[grp].min(), repeated.loc[grp].max()], [i, i], color=c, lw=1.2, alpha=.8, zorder=2)
        ax.scatter(overall.loc[grp, col], i, s=34 if grp == 4 else 20, color=c, zorder=3,
                   edgecolor="white", linewidth=.8)
        ax.text(1.02, i, fmt.format(overall.loc[grp, col]), transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.8, color=INK if grp == 4 else INK2)
    ax.set_yticks(range(len(order)), [f"Group {k}" for k in order])
    for lab in ax.get_yticklabels():
        if lab.get_text() == "Group 4":
            lab.set_color(BLUE)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-.6, len(order) - .4)
    ax.set_title(title); ax.set_xlabel(xlabel)
axs[0].xaxis.set_major_formatter(COMMA)
save(fig, "01_overall")

# Figure 2: all 12 budget/horizon settings, same resource grid for each n.
settings = h[["budget_total", "years"]].drop_duplicates()
settings["annual"] = settings.budget_total / settings.years
settings = settings.sort_values(["annual", "years", "budget_total"])
setting_keys = list(settings[["budget_total", "years"]].itertuples(index=False, name=None))
setting_labels = ["Unlimited / 1y" if np.isinf(b) else f"${int(b)} / {int(y)}y" for b,y in setting_keys]
setting_map = dict(zip(setting_keys, setting_labels))
g4h = g4h.copy()
g4h["setting"] = [setting_map[b,y] for b,y in zip(g4h.budget_total,g4h.years)]
budget = g4h.groupby(["roommates", "setting"]).agg(
    score=("score", "mean"), mismatch=("mismatch_daily", "mean"),
    sockless_pct=("any_sockless", lambda s: 100*s.mean()),
    exhausted_pct=("exhausted", lambda s: 100*s.mean()),
).reset_index()
budget.to_csv(OUT / "budget.csv", index=False)
short = {lab: ("Unlimited\n1 yr" if np.isinf(b) else f"${int(b):,}\n{int(y)} yr")
         for lab, (b, y) in zip(setting_labels, setting_keys)}
fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.3), gridspec_kw={"hspace": .55})
for ax, col, title, fmt, vmax, gamma in [
        (axs[0], "mismatch", "Mismatch embarrassment per roommate-day", ".2f", None, .5),
        (axs[1], "sockless_pct", "Simulations with any sockless day (%)", ".0f", 100, 1)]:
    tab = budget.pivot(index="roommates", columns="setting", values=col).reindex(columns=setting_labels)
    tab.columns = [short[c] for c in tab.columns]
    heatmap(ax, tab, title, fmt, vmax=vmax, gamma=gamma)
    ax.set_ylabel("Roommates")
    ax.tick_params(axis="x", labelsize=8.6)
save(fig,"02_budget")

# Figure 3: positive finite budgets in homogeneous households.
finite = homo[np.isfinite(homo.rho) & homo.rho.gt(0)]
pressure = finite.groupby(["group","rho"]).agg(
    exhausted_pct=("exhausted",lambda s:100*s.mean()),
    exhaustion_time=("exhaustion_fraction","mean"),
    sockless_pct=("sockless_fraction",lambda s:100*s.mean()),
).reset_index()
pressure.to_csv(OUT/"budget_pressure.csv",index=False)
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5), gridspec_kw={"wspace": .42})
for ax, col, title, ylabel in zip(axs, ["exhausted_pct", "exhaustion_time", "sockless_pct"],
        ["Budget runs out", "When it first runs out", "Days without socks"],
        ["Simulations (%)", "Fraction of the run", "Roommate-days (%)"]):
    for grp in sorted(set(pressure.group) - {4}):
        s = pressure[pressure.group == grp].set_index("rho")[col].dropna()
        ax.plot(s.index, s.values, color=GRAY, lw=.8, alpha=.7, zorder=2)
    ours = pressure[pressure.group == 4].set_index("rho")[col].dropna()
    ax.plot(ours.index, ours.values, color=BLUE, lw=1.8, zorder=3)
    ax.scatter(ours.index, ours.values, s=10, color=BLUE, zorder=4)
    ax.set_xscale("log"); ax.set_xlabel(r"Replacement pressure $\rho$")
    ax.set_xticks([3, 10, 30, 100, 300], ["3", "10", "30", "100", "300"]); ax.minorticks_off()
    ax.set_title(title); ax.set_ylabel(ylabel)
handles = [plt.Line2D([], [], color=BLUE, lw=1.8), plt.Line2D([], [], color=GRAY, lw=.8)]
fig.legend(handles, ["Group 4", "Each of the other eight groups"], loc="lower center",
           bbox_to_anchor=(.5, 1.0), ncol=2, handlelength=1.6)
save(fig,"03_exhaustion")

# Figure 4: direct neighbor and majority/minority comparisons.
partners=g4[g4.household_type.isin(["n2_mixed","n9_minority"])].copy()
partners["partner"]=partners.roster_tuple.map(lambda r:next(k for k in r if k!=4))
partners["context"]=np.where(partners.roommates==2,"1 Group 4 + 1 other",
    np.where(partners.role=="minority","1 Group 4 + 8 other","8 Group 4 + 1 other"))
neighbor=partners.groupby(["context","partner"]).agg(
    score=("score","mean"),mismatch=("mismatch_daily","mean"),
    sockless_pct=("any_sockless",lambda s:100*s.mean()),
).reset_index()
neighbor.to_csv(OUT/"neighbors.csv",index=False)
fig, axs = plt.subplots(3, 1, figsize=(6.5, 4.6), gridspec_kw={"hspace": .62})
rows = ["1 Group 4 + 1 other", "1 Group 4 + 8 other", "8 Group 4 + 1 other"]
for ax, col, title, fmt, gamma in [
        (axs[0], "score", "Group 4 total embarrassment per roommate-day", ",.0f", 1),
        (axs[1], "mismatch", "Group 4 mismatch embarrassment per roommate-day", ".2f", 1),
        (axs[2], "sockless_pct", "Simulations with any Group 4 sockless day (%)", ".1f", 1)]:
    tab = neighbor.pivot(index="context", columns="partner", values=col).reindex(rows)
    tab.columns = [f"Group {k}" for k in tab.columns]
    heatmap(ax, tab, title, fmt, gamma=gamma)
axs[2].set_xlabel("The other group in the household")
save(fig,"04_neighbors")

# Figure 5: change capacity within a fixed budget/horizon and household type.
capacity=g4[g4.household_type.isin(["n9_homo","n9_diff","n36_homo","n36_diff"])].groupby(
    ["household_type","roommates","budget_total","years","drawer_multiplier"]).agg(
    score=("score","mean"),sockless_pct=("sockless_fraction",lambda s:100*s.mean())).reset_index()
capacity.to_csv(OUT/"capacity.csv",index=False)
fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.4), gridspec_kw={"hspace": .6, "wspace": .32})
styles = {"n9_homo": ("9, all Group 4", BLUE, "o", "-"), "n9_diff": ("9, mixed", GRAY, "o", "-"),
          "n36_homo": ("36, all Group 4", BLUE, "s", (0, (3, 1.5))), "n36_diff": ("36, mixed", GRAY, "s", (0, (3, 1.5)))}
for row, (b, y) in enumerate([(300, 1), (1500, 10)]):
    for typ, (label, color, marker, ls) in styles.items():
        sub = capacity[(capacity.household_type == typ) & (capacity.budget_total == b) & (capacity.years == y)]
        for ax, col in zip(axs[row], ["score", "sockless_pct"]):
            ax.plot(sub.drawer_multiplier, sub[col], color=color, ls=ls, lw=1.4, marker=marker,
                    ms=4.2, mec="white", mew=.7, label=label)
    for ax in axs[row]:
        ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 10], ["1", "2", "4", "10"]); ax.minorticks_off()
        ax.set_xlabel("Drawer multiplier")
    axs[row, 0].set_yscale("symlog", linthresh=1); axs[row, 0].set_ylim(bottom=0)
    axs[row, 0].yaxis.set_major_formatter(COMMA)
    axs[row, 0].set_ylabel("Total embarrassment / day")
    axs[row, 1].set_ylabel("Sockless roommate-days (%)")
    budget_label = f"\\${b:,} over {y} year" + ("s" if y > 1 else "")
    axs[row, 0].set_title(budget_label)
    axs[row, 1].set_title(budget_label)
axs[0, 1].legend(loc="upper right", fontsize=8.8, handlelength=2.2)
save(fig,"05_capacity")

# Figure 6: paired unit comparison. C differs; this is not a fixed-C experiment.
keys=["group","household_type","roster","role","roommates","drawer_multiplier",
      "budget_total","years","days","seed"]
pair=g[g.unit==4].merge(g[g.unit==5],on=keys,suffixes=("_4","_5"),validate="one_to_one")
assert len(pair)==132192
for col in ["score","mismatch_daily","sockless_fraction"]:
    pair["delta_"+col]=pair[col+"_5"]-pair[col+"_4"]
p4=pair[pair.group==4]
unit=p4.groupby("roommates").agg(
    score4=("score_4","mean"),score5=("score_5","mean"),
    mismatch4=("mismatch_daily_4","mean"),mismatch5=("mismatch_daily_5","mean"),
    delta_score=("delta_score","mean"),delta_mismatch=("delta_mismatch_daily","mean"),
    delta_sockless_pp=("delta_sockless_fraction",lambda s:100*s.mean()),
    pairs=("seed","size"),
)
unit.to_csv(OUT/"unit_comparison.csv")
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True, gridspec_kw={"wspace": .12})
ys = np.arange(len(unit.index))[::-1]
for ax, col, title, xlabel, fmt in zip(axs, ["delta_score", "delta_mismatch", "delta_sockless_pp"],
        ["Total embarrassment", "Mismatch embarrassment", "Days without socks"],
        ["Per roommate-day", "Per roommate-day", "Percentage points"],
        ["{:,.0f}", "{:.2f}", "{:.2f}"]):
    vals = unit[col].to_numpy()
    lo = vals.min()
    ax.barh(ys, vals, height=.62, color=BLUE, zorder=2)
    ax.axvline(0, color=MUTED, lw=.6)
    for yi, v in zip(ys, vals):
        txt = "0" if abs(v) < 5e-3 * abs(lo) else fmt.format(v).replace("-", "\u2212")
        ax.text(min(v, 0) + lo * .03, yi, txt, ha="right", va="center", fontsize=8.4)
    ax.set_xlim(lo * 1.55, -lo * .04)
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_title(title); ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
axs[0].set_yticks(ys, unit.index.astype(str)); axs[0].set_ylabel("Roommates")
axs[0].xaxis.set_major_formatter(COMMA)
save(fig,"06_unit")

# Supplement: replace one of five groups with Group 4, holding the other four fixed.
five=h[h.household_type=="n5"].copy()
five["contains4"]=five.roster_tuple.map(lambda r:4 in r)
records=[]
controls=["unit","drawer_multiplier","budget_total","years","days","seed"]
gfive=g[g.household_type=="n5"]
for other in sorted(set(x.group)-{4}):
    left=five[five.contains4 & ~five.roster_tuple.map(lambda r:other in r)].copy()
    right=five[~five.contains4 & five.roster_tuple.map(lambda r:other in r)].copy()
    left["core"]=left.roster_tuple.map(lambda r:tuple(k for k in r if k!=4))
    right["core"]=right.roster_tuple.map(lambda r:tuple(k for k in r if k!=other))
    matches=left.merge(right,on=["core"]+controls,suffixes=("_4","_other"),validate="one_to_one")
    assert len(matches)==10080
    shared4=gfive[(gfive.group!=4)].groupby("sim_id").agg(score=("score","mean"),mismatch=("mismatch_daily","mean"))
    sharedother=gfive[gfive.group!=other].groupby("sim_id").agg(score=("score","mean"),mismatch=("mismatch_daily","mean"))
    matches=matches.join(shared4,on="sim_id_4",rsuffix="_shared4").join(sharedother,on="sim_id_other",rsuffix="_sharedother")
    records.append({"replaced_group":other,"pairs":len(matches),
        "shared_score_delta":(matches.score-matches.score_sharedother).mean(),
        "shared_mismatch_delta":(matches.mismatch-matches.mismatch_sharedother).mean(),
        "household_sockless_pp":100*(matches.sockless_fraction_4-matches.sockless_fraction_other).mean(),
        "household_spend_delta":(matches.household_spend_per_year_4-matches.household_spend_per_year_other).mean()})
external=pd.DataFrame(records).set_index("replaced_group")
external.to_csv(OUT/"externalities.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": .35})
for ax, col, title in zip(axs, ["shared_score_delta", "shared_mismatch_delta"],
        ["Other four groups: total embarrassment", "Other four groups: mismatch embarrassment"]):
    xs = np.arange(len(external.index))
    ax.bar(xs, external[col], width=.62, color=BLUE, zorder=2)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set_xticks(xs, external.index.astype(str)); ax.grid(axis="x", visible=False)
    ax.set_xlabel("Group replaced by Group 4"); ax.set_title(title); ax.set_ylabel("Change per roommate-day")
save(fig,"07_externalities")

# Figure 8: where Group 4 ranks, and where its gap to the leader comes from.
ROWS = [("n1", "alone", "Alone"), ("n2_homo", "same group", "2, all Group 4"), ("n2_mixed", "mixed", "2, mixed"),
        ("n5", "mixed", "5, mixed"), ("n9_diff", "mixed", "9, one of each group"),
        ("n9_minority", "minority", "9, Group 4 is the 1"), ("n9_minority", "majority", "9, Group 4 is the 8"),
        ("n9_homo", "same group", "9, all Group 4"), ("n18_diff", "mixed", "18, two of each group"),
        ("n18_homo", "same group", "18, all Group 4"), ("n36_diff", "mixed", "36, four of each group"),
        ("n36_homo", "same group", "36, all Group 4")]
cfg = ["household_type", "role", "unit", "drawer_multiplier", "budget_total", "years"]
cell = g.groupby(cfg + ["group"]).score.mean().unstack("group")
ranks = cell.rank(axis=1, method="min")
leader = overall["mean"].drop(4).idxmin()
household_rank = []
for typ, role, label in ROWS:
    sel = (ranks.index.get_level_values(0) == typ) & (ranks.index.get_level_values(1) == role)
    contrib = (g[(g.household_type == typ) & (g.role == role) & (g.group == 4)].score.sum()
               - g[(g.household_type == typ) & (g.role == role) & (g.group == leader)].score.sum()) / len(g4)
    household_rank.append({"household": label, "configurations": int(sel.sum()),
                           **{f"rank_{k}": ranks[sel][k].mean() for k in ranks.columns},
                           "group4_first": int((ranks[sel][4] == 1).sum()),
                           "gap_to_leader": contrib})
household_rank = pd.DataFrame(household_rank).set_index("household")
household_rank.to_csv(OUT / "household_rank.csv")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.55, 1], "wspace": .08})
labels = household_rank.index[::-1]
for i, lab in enumerate(labels):
    if "all Group 4" in lab or "is the 8" in lab:
        for a in (ax, bx):
            a.axhspan(i - .5, i + .5, color=BAND, lw=0, zorder=0)
    others = [household_rank.loc[lab, f"rank_{k}"] for k in ranks.columns if k != 4]
    ax.plot([min(others), max(others)], [i, i], color=RULE, lw=2.2, solid_capstyle="round", zorder=1)
    ax.scatter(others, [i] * len(others), s=15, color=GRAY, lw=0, zorder=2)
    ax.scatter(household_rank.loc[lab, "rank_4"], i, s=40, color=BLUE, edgecolor="white", lw=1, zorder=3)
ax.set_yticks(range(len(labels)), labels); ax.set_ylim(-.6, len(labels) - .4)
ax.set_xlim(.8, 9.2); ax.set_xticks(range(1, 10)); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean rank over configurations (1 = best of 9)")
ax.scatter([], [], s=40, color=BLUE, label="Group 4"); ax.scatter([], [], s=15, color=GRAY, label="Other groups")
ax.legend(loc="lower left", bbox_to_anchor=(-.01, .995), ncol=2, handletextpad=.2, columnspacing=1.2, borderaxespad=0)
gap = household_rank["gap_to_leader"][::-1]
bx.barh(range(len(gap)), gap.clip(lower=0), height=.62, color=BLUE, zorder=2)
for i, v in enumerate(gap):
    txt = "0.00" if abs(v) < .005 else f"{v:+.2f}".replace("-", "\u2212")
    bx.text(max(v, 0) + .35, i, txt, va="center", fontsize=8.4,
            color=INK if v > .5 else INK2)
bx.axvline(0, color=MUTED, lw=.6); bx.grid(axis="y", visible=False)
bx.set_xlim(0, gap.max() * 1.28); bx.tick_params(axis="y", length=0)
bx.set_xlabel(f"Excess over Group {leader} per roommate-day")
bx.set_title(f"Share of the {gap.sum():.1f}-point gap", pad=8)
save(fig, "08_households")

# Figure 9: replaying Group 4's majority households without voluntary discards.
ablation_path = OUT / "discard_ablation.csv"
if ablation_path.exists():
    ab = pd.read_csv(ablation_path)
    checks = ab[ab.variant == "submitted"].merge(g4[["sim_id", "score"]], on="sim_id")
    assert len(checks) and np.allclose(checks.group4_score, checks.score, atol=1e-4)
    off = ab[ab.variant == "no_voluntary_discards"].merge(
        g4[["sim_id", "household_type", "role", "score", "sockless_days", "exhausted", "budget_exhausted_on_day", "days"]],
        on="sim_id", suffixes=("_off", ""), validate="one_to_one")
    finite_sims = g[np.isfinite(g.budget_total)]
    ablation_rows = []
    for typ, role, label in [("n2_homo", "same group", "2, all Group 4"), ("n9_homo", "same group", "9, all Group 4"),
                             ("n9_minority", "majority", "9, Group 4 is the 8"), ("n18_homo", "same group", "18, all Group 4"),
                             ("n36_homo", "same group", "36, all Group 4"), ("n9_minority", "minority", "9, Group 4 is the 1")]:
        sub = off[(off.household_type == typ) & (off.role == role)]
        peers = finite_sims[(finite_sims.household_type == typ) & (finite_sims.role == role) & (finite_sims.group != 4)]
        peer_means = peers.groupby("group").score.mean()
        ablation_rows.append({"household": label, "simulations": len(sub),
            "submitted": sub.score.mean(), "no_voluntary_discards": sub.group4_score.mean(),
            "submitted_sockless_days": sub.sockless_days.mean(), "off_sockless_days": sub.group4_sockless_days.mean(),
            "submitted_exhausted_pct": 100 * sub.exhausted.mean(),
            "off_exhausted_pct": 100 * sub.budget_exhausted_on_day_off.notna().mean(),
            "submitted_rank": 1 + (peer_means < sub.score.mean()).sum(),
            "off_rank": 1 + (peer_means < sub.group4_score.mean()).sum(),
            "peer_best": peer_means.min(), "peer_median": peer_means.median(), "peer_worst": peer_means.max(),
            **{f"peer_{k}": v for k, v in peer_means.items()}})
    ablation_summary = pd.DataFrame(ablation_rows).set_index("household")
    ablation_summary.to_csv(OUT / "discard_ablation_summary.csv")
    # Standings if every replayed simulation had gone as in the replay. The other groups in
    # those households change too, so their scores are replaced as well.
    replay = off.set_index("sim_id")
    cf = g[["sim_id", "group", "score"]].copy()
    in_replay = cf.sim_id.isin(replay.index)
    mine = in_replay & (cf.group == 4)
    cf.loc[mine, "score"] = cf.loc[mine, "sim_id"].map(replay.group4_score).values
    theirs = in_replay & (cf.group != 4)
    cf.loc[theirs, "score"] = cf.loc[theirs, "sim_id"].map(replay.others_score).values
    assert cf.score.notna().all()
    standings = cf.groupby("group").score.mean()
    standings.rename("counterfactual_mean").to_frame().join(overall["mean"]).to_csv(OUT / "discard_ablation_standings.csv")
    ablation_overall = {"replayed_simulations": int(len(off)), "verified_replays": int(len(checks)),
                        "submitted_mean": float(overall.loc[4, "mean"]),
                        "counterfactual_mean": float(standings[4]),
                        "counterfactual_rank": int(standings.rank(method="min")[4]),
                        "counterfactual_leader": int(standings.drop(4).idxmin()),
                        "counterfactual_leader_mean": float(standings.drop(4).min())}
    (OUT / "discard_ablation_overall.json").write_text(json.dumps(ablation_overall, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    labels = ablation_summary.index[::-1]
    def rel(v, lab):
        return 100 * (v / ablation_summary.loc[lab, "peer_median"] - 1)
    for i, lab in enumerate(labels):
        peer = [rel(ablation_summary.loc[lab, f"peer_{k}"], lab) for k in peer_means.index]
        a, b = rel(ablation_summary.loc[lab, "submitted"], lab), rel(ablation_summary.loc[lab, "no_voluntary_discards"], lab)
        ax.scatter(peer, [i] * len(peer), s=14, color=GRAY, lw=0, zorder=2)
        ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=.9,
                    shrinkA=4, shrinkB=4, mutation_scale=8), zorder=3)
        ax.scatter(a, i, s=40, color=BLUE, edgecolor="white", lw=1, zorder=4)
        ax.scatter(b, i, s=40, color=ORANGE, edgecolor="white", lw=1, zorder=4)
    ax.axvline(0, color=MUTED, lw=.6)
    ax.set_yticks(range(len(labels)), labels); ax.grid(axis="y", visible=False)
    ax.set_ylim(-.6, len(labels) - .4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "\u2212").replace("+0%", "0%")))
    ax.set_xlabel("Group 4 embarrassment relative to the median other group in the same households")
    ax.scatter([], [], s=40, color=BLUE, label="Submitted player")
    ax.scatter([], [], s=40, color=ORANGE, label="Same player, no voluntary discards")
    ax.scatter([], [], s=14, color=GRAY, label="Other groups")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, handletextpad=.2, columnspacing=1.2)
    save(fig, "09_discard_ablation")

g.groupby(["group","household_type","role"]).agg(
    sims=("sim_id","size"),score=("score","mean"),mismatch=("mismatch_daily","mean"),
    sockless_pct=("any_sockless",lambda s:100*s.mean()),exhausted_pct=("exhausted",lambda s:100*s.mean()),
).to_csv(OUT/"household_summary.csv")
metrics={
    "simulations":len(h),"rows":len(x),"configurations":len(h)//3,"faults":int(x.faults.sum()),
    "group4_sockless_score_share":float(1-g4.mismatch_daily.sum()/g4.score.sum()),
    "group4_pairs":len(p4),
    "group4_unit4_score":float(p4.score_4.mean()),"group4_unit5_score":float(p4.score_5.mean()),
    "group4_unit4_mismatch":float(p4.mismatch_daily_4.mean()),"group4_unit5_mismatch":float(p4.mismatch_daily_5.mean()),
    "group4_unit4_sockless_pct":float(100*p4.sockless_fraction_4.mean()),
    "group4_unit5_sockless_pct":float(100*p4.sockless_fraction_5.mean()),
    "group4_homo_exhaustion_pct":float(100*g4h.exhausted.mean()),
    "group4_homo_sockless_pct":float(100*g4h.any_sockless.mean()),
    "group4_homo_no_sockless_after_exhaustion_pct":float(100*((g4h.exhausted)&~g4h.any_sockless).mean()),
}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
print(json.dumps(metrics,indent=2))
print(overall.round(4).to_string())
