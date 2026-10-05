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

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def heatmap(fig, ax, data, title, label, vmax=None, fmt=".1f"):
    im = ax.imshow(data.to_numpy(), aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(data.index)), data.index)
    ax.set_title(title)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            color = "white" if val < im.norm.vmax * .45 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label=label, shrink=.85)


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
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3))
for ax, col, title in zip(axs, ["mean", "mismatch_per_dressed_day"],
                         ["Total embarrassment", "Mismatch embarrassment"]):
    repeated = seed_summary[col].unstack()
    bounds = np.array([overall[col] - repeated.min(axis=1), repeated.max(axis=1) - overall[col]])
    ax.bar(overall.index.astype(str), overall[col], yerr=bounds, capsize=2)
    ax.set_xlabel("Group")
    ax.set_ylabel("Per roommate-day" if col == "mean" else "Per dressed roommate-day")
    ax.set_title(title)
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
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8))
for ax,col,title,label,fmt in [
    (axs[0],"mismatch","Mismatch embarrassment","Per roommate-day",".2f"),
    (axs[1],"sockless_pct","Simulations with sockless days","Percent of simulations",".0f")]:
    tab = budget.pivot(index="roommates",columns="setting",values=col).reindex(columns=setting_labels)
    heatmap(fig,ax,tab,title,label,vmax=100 if col=="sockless_pct" else None,fmt=fmt)
    ax.set_ylabel("Roommates (all Group 4)")
save(fig,"02_budget")

# Figure 3: positive finite budgets in homogeneous households.
finite = homo[np.isfinite(homo.rho) & homo.rho.gt(0)]
pressure = finite.groupby(["group","rho"]).agg(
    exhausted_pct=("exhausted",lambda s:100*s.mean()),
    exhaustion_time=("exhaustion_fraction","mean"),
    sockless_pct=("sockless_fraction",lambda s:100*s.mean()),
).reset_index()
pressure.to_csv(OUT/"budget_pressure.csv",index=False)
fig,axs=plt.subplots(1,3,figsize=(7.2,3.3))
for ax,col,title,ylabel in zip(axs,["exhausted_pct","exhaustion_time","sockless_pct"],
    ["Replacement purchase fails","When purchase first fails","Sockless roommate-days"],
    ["Percent of simulations","Fraction of horizon (conditional)","Percent of roommate-days"]):
    peers=pressure[pressure.group!=4].groupby("rho")[col].median()
    ours=pressure[pressure.group==4].set_index("rho")[col]
    ax.plot(peers.index,peers.values,".-",label="Median of other groups")
    ax.plot(ours.index,ours.values,".-",label="Group 4")
    ax.set_xscale("log"); ax.set_xlabel(r"Replacement pressure $\rho$")
    ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel)
axs[0].legend(fontsize=8)
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
fig,axs=plt.subplots(3,1,figsize=(7.2,4.8))
for ax,col,title,label,fmt in [
    (axs[0],"score","Group 4 total embarrassment","Per roommate-day",".0f"),
    (axs[1],"mismatch","Group 4 mismatch embarrassment","Per roommate-day",".2f"),
    (axs[2],"sockless_pct","Simulations with any Group 4 sockless days","Percent of simulations",".1f")]:
    tab=neighbor.pivot(index="context",columns="partner",values=col).reindex(
        ["1 Group 4 + 1 other","1 Group 4 + 8 other","8 Group 4 + 1 other"])
    heatmap(fig,ax,tab,title,label,fmt=fmt)
    ax.set_xlabel("Other group")
    ax.tick_params(axis="x",labelrotation=0)
save(fig,"04_neighbors")

# Figure 5: change capacity within a fixed budget/horizon and household type.
capacity=g4[g4.household_type.isin(["n9_homo","n9_diff","n36_homo","n36_diff"])].groupby(
    ["household_type","roommates","budget_total","years","drawer_multiplier"]).agg(
    score=("score","mean"),sockless_pct=("sockless_fraction",lambda s:100*s.mean())).reset_index()
capacity.to_csv(OUT/"capacity.csv",index=False)
fig,axs=plt.subplots(2,2,figsize=(7.2,4.4))
for row,(b,y) in enumerate([(300,1),(1500,10)]):
    for typ,label in [("n9_homo","9, all Group 4"),("n9_diff","9, mixed"),
                      ("n36_homo","36, all Group 4"),("n36_diff","36, mixed")]:
        sub=capacity[(capacity.household_type==typ)&(capacity.budget_total==b)&(capacity.years==y)]
        for ax,col in zip(axs[row],["score","sockless_pct"]):
            ax.plot(sub.drawer_multiplier,sub[col],"o-",label=label)
            ax.set_xscale("log"); ax.set_xticks([1,2,4,10],["1","2","4","10"])
            ax.set_xlabel("Drawer multiplier")
    axs[row,0].set_ylabel("Total embarrassment / day")
    axs[row,0].set_yscale("symlog",linthresh=1)
    axs[row,0].set_ylim(bottom=0)
    axs[row,1].set_ylabel("Sockless roommate-days (%)")
    for ax in axs[row]: ax.set_title(f"${b} over {y} year"+("s" if y>1 else ""))
axs[0,0].legend(fontsize=8)
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
fig,axs=plt.subplots(1,3,figsize=(7.2,3.3))
for ax,col,title,ylabel in zip(axs,["delta_score","delta_mismatch","delta_sockless_pp"],
    ["Total embarrassment","Mismatch embarrassment","Sockless roommate-days"],
    ["Change per roommate-day","Change per roommate-day","Change (percentage points)"]):
    ax.bar(unit.index.astype(str),unit[col]); ax.axhline(0,color="black",linewidth=.8)
    ax.set_xlabel("Roommates"); ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel)
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
fig,axs=plt.subplots(1,2,figsize=(7.2,3))
for ax,col,title,ylabel in zip(axs,["shared_score_delta","shared_mismatch_delta"],
    ["Other four groups: total embarrassment","Other four groups: mismatch embarrassment"],
    ["Change per roommate-day","Change per roommate-day"]):
    ax.bar(external.index.astype(str),external[col]); ax.axhline(0,color="black",linewidth=.8)
    ax.set_xlabel("Group replaced by Group 4"); ax.set_title(title,fontsize=10); ax.set_ylabel(ylabel)
save(fig,"07_externalities")

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
