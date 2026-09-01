import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

from hyppo.discrim import DiscrimTwoSample

from AFQ.viz.utils import COLOR_DICT

OLD_COLORS = {"Left": "#6E6E6E", "Right": "#B4B4B4"}

N_PERM      = int(1e4)
ALPHA       = 0.05
RANDOM_SEED = 42


DATASETS = [
    ("HCP", "compiled_trt_results.csv", "compiled_wdsc_hcp.csv"),
    ("HBN",         "compiled_hbn_results.csv", "compiled_wdsc_hbn.csv"),
]

BUNDLE_GROUPS = [
    {
        "name": "VOF",
        "old_name": "Vertical Occipital",
        "subbundles": [
            ("Early Visual", "EV"),
            ("Posterior Vertical Occipital", "pVOF"),
            ("Anterior Vertical Occipital", "aVOF"),
        ],
    },
    {
        "name": "pAF",
        "old_name": "Posterior Arcuate",
        "subbundles": [
            ("Posterior Arcuate", "pAF"),
            ("Temporo-parietal", "TP"),
        ],
    },
]

DWI_METRICS = ["dti_fa", "dti_md"]
METRICS = DWI_METRICS + ["wdsc"]
METRIC_LABELS = {"dti_fa": "FA", "dti_md": "MD", "wdsc": "wDSC"}
YTICKS = {"dti_fa": [0.7, 0.8, 0.9, 1.0],
          "dti_md": [0.7, 0.8, 0.9, 1.0],
          "wdsc":   [0.9, 0.95, 1.0]}

HEMIS = ("Left", "Right")


FONT_SIZE = 18

SIG_PAD  = 0.02
SIG_STEP = 0.07
SIG_LW   = 1.2

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica", "Helvetica Neue", "TeX Gyre Heros",
    "Nimbus Sans", "Arial", "Liberation Sans", "DejaVu Sans",
]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica"
plt.rcParams["mathtext.it"] = "Helvetica:italic"
plt.rcParams["mathtext.bf"] = "Helvetica:bold"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

plt.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE * 1.3,
    "axes.labelsize":   FONT_SIZE * 1.1,
    "xtick.labelsize":  FONT_SIZE * 0.8,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE,
    "figure.titlesize": FONT_SIZE * 1.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "grid.color": "#CCCCCC",
    "grid.linewidth": 0.7,
})

def _new_color(hemi, group, suf, i):
    if f"{hemi} {suf}" in COLOR_DICT:
        return COLOR_DICT[f"{hemi} {suf}"]
    base = np.asarray(COLOR_DICT[f"{hemi} {group['old_name']}"][:3], float)
    return tuple(np.clip(base * (1.0 - 0.25 * i), 0, 1))


def _draw_sig_bracket(ax, x_new, y_new, x_old, y_old, y_top, higher, pad):
    ls = "-" if higher else ":"
    ax.plot([x_new, x_new, x_old, x_old],
            [y_new + pad, y_top, y_top, y_old + pad],
            ls=ls, lw=SIG_LW, color="black", solid_capstyle="butt",
            clip_on=False, zorder=5)


def _build_arr(df, pipe, tract, metric):
    sub = df[(df["Pipeline"] == pipe) & (df["tractID"] == tract)][
        ["Subject", "Session", "nodeID", metric]]
    n = sub["Subject"].nunique()
    p = sub["nodeID"].nunique()
    k = sub["Session"].nunique()
    return (sub.sort_values(["Subject", "nodeID", "Session"])[metric]
               .to_numpy().reshape(n, p, k))


def _dist(arr):
    """(n, p, k) -> X (n*k, p), y (n*k,) subject labels."""
    n, _, k = arr.shape
    X = arr.transpose(0, 2, 1).reshape(n * k, -1)
    y = np.repeat(np.arange(n), k)  # subject labels
    return pairwise_distances(X, metric="euclidean"), y


def group_discrim_tests(df, dataset, group, metric, hemi):
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
            "dataset": dataset,
            "group": group["name"], "metric": metric, "hemi": hemi,
            "subbundle": suf, "sub_label": label,
            "n_subjects": n, "n_sessions": k, "n_measurements": n * k,
            "discrim_new": d_new, "discrim_old": d_old,
            "discrim_delta": d_new - d_old,
            "discrim_pval": pval,
        })
    return rows


def group_wdsc_rows(wdsc, dataset, group, hemi):
    """Mean weighted-Dice per sub-bundle vs its source bundle, paired by subject."""
    d = wdsc[wdsc["hemisphere"] == hemi].set_index(["subject", "bundle"])
    old = d[d["technique"] == "old"]["weighted_dice"]
    new = d[d["technique"] == "new"]["weighted_dice"]

    rows = []
    for suf, label in group["subbundles"]:
        pair = pd.concat([new.xs(suf, level="bundle").rename("new"),
                          old.xs(suf, level="bundle").rename("old")],
                         axis=1, join="inner")
        diff = pair["new"] - pair["old"]
        new_val, old_val = pair["new"].mean(), pair["old"].mean()
        pval = wilcoxon(diff, alternative="two-sided").pvalue
        rows.append({
            "dataset": dataset,
            "group": group["name"], "metric": "wdsc", "hemi": hemi,
            "subbundle": suf, "sub_label": label,
            "n_subjects": len(pair), "n_sessions": 2,
            "n_measurements": len(pair),
            "discrim_new": new_val, "discrim_old": old_val,
            "discrim_delta": diff.mean(),
            "discrim_pval": pval,
        })
    return rows


def run_analysis(output_dir, out_fname):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tests = (sum(len(g["subbundles"]) for g in BUNDLE_GROUPS)
               * len(DWI_METRICS) * len(HEMIS) * len(DATASETS))
    print(f"Metrics    : {METRICS}")
    print(f"Datasets   : {[d for d, _, _ in DATASETS]}")
    print(f"Group-level: {n_tests} tests (dataset x hemi x sub-bundle x metric)\n")

    grecs = []
    for dataset, csv_path, wdsc_path in DATASETS:
        df = pd.read_csv(csv_path)
        wdsc = pd.read_csv(wdsc_path)
        for group in BUNDLE_GROUPS:
            for metric in DWI_METRICS:
                for hemi in HEMIS:
                    print(f"  {dataset} {group['name']} [{metric}] {hemi}"
                          + " " * 20, end="\r")
                    grecs.extend(
                        group_discrim_tests(df, dataset, group, metric, hemi))
            for hemi in HEMIS:
                grecs.extend(group_wdsc_rows(wdsc, dataset, group, hemi))
    print()

    group_results = pd.DataFrame(grecs)
    group_results["discrim_sig"] = (
        group_results.groupby(["dataset", "metric"])["discrim_pval"]
        .transform(lambda p: multipletests(p, method="holm", alpha=ALPHA)[0]))

    print_stats(group_results)

    group_results.to_csv(output_dir / "discriminability_by_group.csv", index=False)

    plot_combined(group_results, output_dir, out_fname)
    return group_results


def _layout_groups(groups, old_offset=0.9,
                   sub_spacing=0.8, hemi_gap=2.0, group_gap=1.6, **_):
    layout, hemi_dividers, group_dividers, cursor = [], [], [], 0.0
    for gi, group in enumerate(groups):
        g_entry, left_last = {}, None
        n_sub = len(group["subbundles"])
        for hemi in HEMIS:
            sub_xs = [cursor + i * sub_spacing for i in range(n_sub)]
            old_x = (sub_xs[-1] if sub_xs else cursor) + old_offset
            g_entry[hemi] = {"old_x": old_x, "sub_xs": sub_xs}
            if hemi == "Left":
                left_last = old_x
                cursor = old_x + hemi_gap
            else:
                hemi_dividers.append((left_last + sub_xs[0]) / 2)  # first *new* bar now
                cursor = old_x + group_gap
        layout.append(g_entry)
        if gi < len(groups) - 1:
            group_dividers.append(cursor - group_gap / 2)
    return layout, hemi_dividers, group_dividers, (-0.6, cursor - group_gap + 0.6)


def _rows_for(group_results, dataset, group_name, metric, hemi):
    return group_results[(group_results["dataset"] == dataset) &
                         (group_results["group"] == group_name) &
                         (group_results["metric"] == metric) &
                         (group_results["hemi"] == hemi)]


def _draw_panel(ax, group_results, dataset, metric, groups, stat="discrim",
                show_ylabel=True, show_xlabels=True, show_title=True,
                show_yticklabels=True):
    bar_width = 0.62
    layout, hemi_dividers, group_dividers, xlim = _layout_groups(groups)
    xtick_positions, xtick_labels = [], []
    ticks = YTICKS[metric]
    span = ticks[-1] - ticks[0]
    pad = SIG_PAD * span
    step = SIG_STEP * span

    for group, g_layout in zip(groups, layout):
        for hemi in HEMIS:
            hemi_ = hemi[0].upper()
            rows = _rows_for(group_results, dataset, group["name"], metric, hemi)
            sig_pairs = []

            for i, ((suf, label), xi) in enumerate(
                    zip(group["subbundles"], g_layout[hemi]["sub_xs"])):
                r = rows[rows["subbundle"] == suf]
                r = r.iloc[0] if len(r) else None
                v = r[f"{stat}_new"] if r is not None else 0

                ax.bar(xi, v, width=bar_width,
                       color=_new_color(hemi, group, suf, i),
                       edgecolor="white", linewidth=0.8, zorder=3)
                if r is not None and bool(r[f"{stat}_sig"]):
                    sig_pairs.append((xi, v, r[f"{stat}_delta"] > 0))
                xtick_positions.append(xi)
                xtick_labels.append(hemi_ + " " + label)

            old_x = g_layout[hemi]["old_x"]
            old_val = rows[f"{stat}_old"].iloc[0] if len(rows) else 0
            ax.bar(old_x, old_val, width=bar_width, color=OLD_COLORS[hemi],
                   edgecolor="white", linewidth=0.8, zorder=3)
            xtick_positions.append(old_x)
            xtick_labels.append(hemi_ + " Old")

            y_cursor = None
            for xi, v, higher in sorted(sig_pairs, key=lambda t: -t[0]):
                y_top = max(v, old_val) + pad + 0.35 * step
                if y_cursor is not None:
                    y_top = max(y_top, y_cursor + step)
                y_cursor = y_top
                _draw_sig_bracket(ax, xi, v, old_x, old_val, y_top, higher, pad)

    for dv in hemi_dividers:
        ax.axvline(dv, color="#BBBBBB", lw=0.8, ls=(0, (3, 3)), zorder=1)
    for dv in group_dividers:
        ax.axvline(dv, color="#666666", lw=1.0, zorder=1)

    ax.set_xlim(*xlim)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels if show_xlabels else [],
                       rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(ticks)
    if not show_yticklabels:
        ax.set_yticklabels([])
    if show_ylabel:
        ax.set_ylabel("Discriminability" if metric in DWI_METRICS else "wDSC")
    if show_title:
        ax.set_title(METRIC_LABELS.get(metric, metric), fontweight="bold", pad=40)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=3, color="#888888")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def plot_combined(group_results, output_dir, out_fname, groups=BUNDLE_GROUPS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, xlim = _layout_groups(groups)
    n_metrics = len(METRICS)
    n_rows = len(DATASETS)
    fig_width = max(16, (xlim[1] - xlim[0]) * 1.15) * (n_metrics / 2.6)
    fig, axes = plt.subplots(n_rows, n_metrics,
                             figsize=(fig_width, 4.4 * n_rows + 1.4),
                             squeeze=False)

    for row_idx, (dataset, _, _) in enumerate(DATASETS):
        for col_idx, metric in enumerate(METRICS):
            new_scale = col_idx == 0 or YTICKS[metric] != YTICKS[METRICS[col_idx - 1]]
            _draw_panel(axes[row_idx, col_idx], group_results, dataset, metric,
                        groups,
                        show_ylabel=new_scale,
                        show_xlabels=(row_idx == n_rows - 1),
                        show_title=(row_idx == 0),
                        show_yticklabels=new_scale)
        axes[row_idx, 0].text(-0.17, 0.5, dataset, rotation=90,
                              ha="center", va="center", fontweight="bold",
                              fontsize=FONT_SIZE * 1.15,
                              transform=axes[row_idx, 0].transAxes)

    plt.tight_layout(h_pad=3.0)
    path = output_dir / out_fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def print_stats(group_results, stat="discrim"):
    print(f"\n{'dataset':<14}{'group':<8}{'metric':<8}{'hemi':<7}"
          f"{'sub-bundle':<30}{'new':>7}{'old':>8}{'delta':>9}{'p':>9}  sig")
    print("-" * 104)
    for dataset, _, _ in DATASETS:
        for group in BUNDLE_GROUPS:
            for metric in METRICS:
                for hemi in HEMIS:
                    rows = _rows_for(group_results, dataset, group["name"],
                                     metric, hemi)
                    for _, r in rows.iterrows():
                        pv = r[f"{stat}_pval"]
                        print(f"{r['dataset']:<14}{r['group']:<8}"
                              f"{METRIC_LABELS.get(metric, metric):<8}"
                              f"{r['hemi']:<7}{r['subbundle']:<30}"
                              f"{r[f'{stat}_new']:>7.3f}{r[f'{stat}_old']:>8.3f}"
                              f"{r[f'{stat}_delta']:>+9.3f}"
                              f"{'      n/a' if pd.isna(pv) else f'{pv:>9.4f}'}"
                              f"  {'*' if bool(r[f'{stat}_sig']) else ''}")
        print()
    tested = group_results[f"{stat}_pval"].notna()
    n_sig = int(group_results[f"{stat}_sig"].sum())
    print("-" * 104)
    print(f"{n_sig}/{int(tested.sum())} significant "
          f"(Holm, alpha={ALPHA})\n")

for dataset, csv_path, _ in DATASETS:
    df = pd.read_csv(csv_path)
    print(f"\n{dataset}:")
    for utid in df["tractID"].unique():
        print(utid, df[df["tractID"] == utid].shape[0])

run_analysis(Path("results"), "combined_discriminability.png")
