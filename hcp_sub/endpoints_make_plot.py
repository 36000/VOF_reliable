import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

df = pd.read_csv("../bundle_dice_analysis.csv")
print("N Subjects:", df["Subject"].nunique())

agg = (
    df.groupby(["Surface", "Bundle", "VisualRegion"])["Proportion"]
    .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    .reset_index()
)

surfaces = agg["Surface"].unique()
bundles = ["Early Visual", "Posterior Vertical Occipital", "Anterior Vertical Occipital"]

both_regions = ["V1", "V2", "V3"]
dorsal_only_regions = ["V3a", "V3b", "LO1", "IPS0"]
ventral_only_regions = ["hV4", "VO1", "VO2"]

surfaces = list(agg["Surface"].unique())
region_sets = {
    surf: both_regions + (dorsal_only_regions if surf == "Dorsal" else ventral_only_regions)
    for surf in surfaces
}

reds = ["#a50f15", "#de2d26", "#fb6a4a"]
blues = ["#08519c", "#3182bd", "#6baed6"]
greens = ["#006d2c", "#31a354", "#74c476"]

region_colors = {
    "V1": reds[0], "V2": reds[1], "V3": reds[2],
    "V3a": blues[0], "V3b": blues[1], "hV4": blues[0], "VO1": blues[1],
    "VO2": blues[2], "LO1": blues[2], "IPS0": greens[2],
}

fig, axes = plt.subplots(
    1, len(surfaces), figsize=(7, 1.8),
    sharey=True, constrained_layout=True
)
if len(surfaces) == 1:
    axes = [axes]

y = np.arange(len(bundles))
bundle_labels = ["Early\nVisual", "Posterior\nVOF", "Anterior\nVOF"]

for ax, surf in tqdm(zip(axes, surfaces)):
    sub = agg[agg["Surface"] == surf]

    lefts = np.zeros(len(bundles))
    for region in region_sets[surf]:
        grp = sub[sub["VisualRegion"] == region].set_index("Bundle").reindex(bundles)
        means = grp["mean"].fillna(0).values
        sems = grp["sem"].fillna(0).values
        print(f"Surface: {surf}, Region: {region}, SEM: {sems}")

        ax.barh(
            y, means,
            height=0.7,
            left=lefts,
            color=region_colors[region],
            edgecolor="white",
            linewidth=0.5,
        )
        for yi, (m, l) in enumerate(zip(means, lefts)):
            if surf == "Dorsal":
                m_thresh = 0.06
            else:
                m_thresh = 0.04

            if m > m_thresh:
                if region == "VO2":
                    ax.text(
                        l + 2.5 * m, yi, "← " + region,
                        ha="center", va="center",
                        color="black", fontsize=7, fontweight="bold",
                    )
                else:
                    ax.text(
                        l + m / 2, yi, region,
                        ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold",
                    )
        lefts += means

    ax.set_yticks(y)
    ax.set_yticklabels(bundle_labels, fontsize=8)
    ax.set_title(f"{surf} Streamline Endpoints", fontsize=8, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

plt.savefig("endpoints_bar.png", dpi=300, bbox_inches="tight")
