import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests

from hyppo.discrim import DiscrimTwoSample


N_PERM      = 1000
ALPHA       = 0.05
RANDOM_SEED = 42

OLD_COLORS = {"Left": "#7F77DD", "Right": "#534AB7"}
NEW_COLORS = {"Left": "#1D9E75", "Right": "#0F6E56"}

BUNDLE_GROUPS = [
    {
        "name": "VOF",
        "old_name": "Vertical Occipital",
        "subbundles": [
            ("Early Visual", "Early V"),
            ("Posterior Vertical Occipital", "post VOF"),
            ("Anterior Vertical Occipital", "ant VOF"),
        ],
    },
    {
        "name": "Posterior Arcuate",
        "old_name": "Posterior Arcuate",
        "subbundles": [
            ("Posterior Arcuate", "Post Arc"),
            ("Temporo-parietal", "Temp Par"),
        ],
    },
]

METRICS = ["dti_fa", "dti_md"]
METRIC_LABELS = {"dti_fa": "FA", "dti_md": "MD"}

HEMIS = ("Left", "Right")


def _build_arr(df, pipe, tract, metric):
    sub = df[(df["Pipeline"] == pipe) & (df["tractID"] == tract)][
        ["Subject", "Session", "nodeID", metric]]
    n = sub["Subject"].nunique()
    p = sub["nodeID"].nunique()
    k = sub["Session"].nunique()
    return (sub.sort_values(["Subject", "nodeID", "Session"])[metric]
               .to_numpy().reshape(n, p, k))


def _to_measurements(arr):
    """(n, p, k) -> X (n*k, p), y (n*k,) subject labels."""
    n, _, k = arr.shape
    X = arr.transpose(0, 2, 1).reshape(n * k, -1)
    y = np.repeat(np.arange(n), k)  # subject labels
    return X, y


def _dist(arr):
    """Euclidean distance matrix over all subject-session measurements."""
    X, y = _to_measurements(arr)
    return pairwise_distances(X, metric="euclidean"), y


def group_discrim_tests(df, group, metric, hemi):
    """Each new sub-bundle vs the old bundle it was split out of."""
    old_arr = _build_arr(df, "old", f"{hemi} {group['old_name']}", metric)
    n, _, k = old_arr.shape
    D_old, y = _dist(old_arr)

    rows = []
    for suf, label in group["subbundles"]:
        D_new, _ = _dist(_build_arr(df, "new", f"{hemi} {suf}", metric))

        np.random.seed(RANDOM_SEED)
        d_new, d_old, pval = DiscrimTwoSample(is_dist=True).test(
            D_new, D_old, y, reps=N_PERM, alt="neq", workers=1)

        rows.append({
            "group": group["name"], "metric": metric, "hemi": hemi,
            "subbundle": suf, "sub_label": label,
            "n_subjects": n, "n_sessions": k, "n_measurements": n * k,
            "discrim_new": d_new, "discrim_old": d_old,
            "discrim_delta": d_new - d_old,
            "discrim_pval": pval,
        })
    return rows


def run_analysis(df, output_dir, out_fname):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tests = (sum(len(g["subbundles"]) for g in BUNDLE_GROUPS)
               * len(METRICS) * len(HEMIS))
    print(f"Metrics    : {METRICS}")
    print(f"Group-level: {n_tests} tests (hemi x sub-bundle x metric)\n")

    grecs = []
    for group in BUNDLE_GROUPS:
        for metric in METRICS:
            for hemi in HEMIS:
                print(f"  {group['name']} [{metric}] {hemi}" + " " * 20, end="\r")
                grecs.extend(group_discrim_tests(df, group, metric, hemi))
    print()

    group_results = pd.DataFrame(grecs)
    group_results["discrim_sig"] = multipletests(
        group_results["discrim_pval"], method="holm", alpha=ALPHA)[0]

    group_results.to_csv(output_dir / "discriminability_by_group.csv", index=False)

    plot_combined(group_results, output_dir, out_fname)
    return group_results


def _layout_groups(groups, bar_width=0.55, old_offset=0.9,
                   sub_spacing=0.8, hemi_gap=2.0, group_gap=1.6):
    layout, hemi_dividers, group_dividers, cursor = [], [], [], 0.0
    for gi, group in enumerate(groups):
        g_entry, left_last = {}, None
        n_sub = len(group["subbundles"])
        for hemi in HEMIS:
            old_x = cursor
            sub_xs = [old_x + old_offset + i * sub_spacing for i in range(n_sub)]
            g_entry[hemi] = {"old_x": old_x, "sub_xs": sub_xs}
            last_x = sub_xs[-1] if sub_xs else old_x
            if hemi == "Left":
                left_last = last_x
                cursor = last_x + hemi_gap
            else:
                hemi_dividers.append((left_last + old_x) / 2)
                cursor = last_x + group_gap
        layout.append(g_entry)
        if gi < len(groups) - 1:
            group_dividers.append(cursor - group_gap / 2)
    return layout, hemi_dividers, group_dividers, (-0.6, cursor - group_gap + 0.6)


def _rows_for(group_results, group_name, metric, hemi):
    return group_results[(group_results["group"] == group_name) &
                         (group_results["metric"] == metric) &
                         (group_results["hemi"] == hemi)]


def _draw_panel(ax, group_results, metric, groups, stat="discrim"):
    bar_width = 0.55
    layout, hemi_dividers, group_dividers, xlim = _layout_groups(
        groups, bar_width=bar_width)
    xtick_positions, xtick_labels = [], []

    for group, g_layout in zip(groups, layout):
        for hemi in HEMIS:
            rows = _rows_for(group_results, group["name"], metric, hemi)
            old_x = g_layout[hemi]["old_x"]
            old_val = rows[f"{stat}_old"].iloc[0] if len(rows) else np.nan

            ax.bar(old_x, old_val if not np.isnan(old_val) else 0,
                   width=bar_width, color=OLD_COLORS[hemi], zorder=3)
            if not np.isnan(old_val):
                ax.text(old_x, old_val + 0.008, f"{old_val:.2f}", ha="center",
                        va="bottom", fontsize=7.5, color=OLD_COLORS[hemi])
            xtick_positions.append(old_x)
            xtick_labels.append(f"Old\n{hemi[0]}")

            for (suf, label), xi in zip(group["subbundles"], g_layout[hemi]["sub_xs"]):
                r = rows[rows["subbundle"] == suf]
                r = r.iloc[0] if len(r) else None
                v = r[f"{stat}_new"] if r is not None else np.nan

                ax.bar(xi, v if not np.isnan(v) else 0, width=bar_width,
                       color=NEW_COLORS[hemi], zorder=3)
                if not np.isnan(v):
                    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=7, color=NEW_COLORS[hemi])
                if r is not None:
                    d = r[f"{stat}_delta"]
                    p = r[f"{stat}_pval"]
                    s = bool(r[f"{stat}_sig"])
                    ax.text(xi, 1.02,
                            f"$\\Delta$={d:+.3f}\np={p:.3f}{'*' if s else ''}",
                            ha="center", va="bottom", fontsize=6,
                            fontweight="bold" if s else "normal",
                            linespacing=1.4,
                            transform=ax.get_xaxis_transform())
                xtick_positions.append(xi)
                xtick_labels.append(label)

    for dv in hemi_dividers:
        ax.axvline(dv, color="gray", lw=0.8, ls="--", alpha=0.5)
    for dv in group_dividers:
        ax.axvline(dv, color="black", lw=1.0, ls="-", alpha=0.35)

    ax.set_xlim(*xlim)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=8.5)
    yticks = [0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=8)
    ax.set_ylabel("Discriminability", fontsize=9)
    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12,
                 fontweight="bold", pad=28)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for group, g_layout in zip(groups, layout):
        for hemi in HEMIS:
            hemi_x = [g_layout[hemi]["old_x"]] + g_layout[hemi]["sub_xs"]
            ax.text(np.mean(hemi_x), -0.19, f"{hemi.lower()} hemi", ha="center",
                    va="top", fontsize=8, color="gray",
                    transform=ax.get_xaxis_transform())

        group_all_x = ([g_layout["Left"]["old_x"], g_layout["Right"]["old_x"]]
                       + g_layout["Left"]["sub_xs"] + g_layout["Right"]["sub_xs"])
        ax.text(np.mean(group_all_x), -0.28, group.get("name", ""), ha="center",
                va="top", fontsize=8.5, color="black", fontweight="bold",
                transform=ax.get_xaxis_transform())


def plot_combined(group_results, output_dir, out_fname, groups=BUNDLE_GROUPS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, xlim = _layout_groups(groups)
    n_metrics = len(METRICS)
    fig_width = max(16, (xlim[1] - xlim[0]) * 1.15) * (n_metrics / 2)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, 6.0), squeeze=False)

    for col_idx, metric in enumerate(METRICS):
        _draw_panel(axes[0, col_idx], group_results, metric, groups)

    fig.suptitle("Test-Retest Discriminability: Old vs New Sub-Bundles\n"
                 "(hyppo discriminability on full tract profiles; "
                 "\u0394 = sub-bundle \u2212 old bundle)",
                 fontsize=14, fontweight="bold", y=1.08)

    legend_handles = [
        mpatches.Patch(color=OLD_COLORS["Left"], label="Old bundle — left hemi"),
        mpatches.Patch(color=OLD_COLORS["Right"], label="Old bundle — right hemi"),
        mpatches.Patch(color=NEW_COLORS["Left"], label="New sub-bundle — left hemi"),
        mpatches.Patch(color=NEW_COLORS["Right"], label="New sub-bundle — right hemi"),
        plt.Line2D([0], [0], linestyle="none", marker="$*$", color="black",
                   markersize=10,
                   label=f"* p < {ALPHA}, Holm over {len(group_results)} tests"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.16))

    plt.tight_layout()
    path = output_dir / out_fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined plot → {path}")


df = pd.read_csv("../compiled_hbn_results.csv")
for utid in df["tractID"].unique():
    print(utid, df[df["tractID"] == utid].shape[0])

run_analysis(df, "./", "combined_discrim_hbn.png")