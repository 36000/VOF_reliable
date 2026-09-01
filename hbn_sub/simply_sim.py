import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from hyppo.discrim import DiscrimOneSample, DiscrimTwoSample
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


PIPELINE = "new"
METRICS  = ["dti_fa", "dti_md"]
METRIC_LABELS = {"dti_fa": "FA", "dti_md": "MD"}
SES1 = "ses-ses1"
SES2 = "ses-ses2"

TRACT_LABELS = {
    # "Left Optic Radiation": "lOR",
    "Left Arcuate": "L AF",
    "Left Posterior Arcuate": "L pAF",
    "Left Temporo-parietal": "L TP",
    "Left Anterior Vertical Occipital": "L aVOF",
    "Left Posterior Vertical Occipital": "L pVOF",
    "Left Early Visual": "L EV",
    # "Right Optic Radiation": "rOR",
    "Right Arcuate": "R AF",
    "Right Posterior Arcuate": "R pAF",
    "Right Temporo-parietal": "R TP",
    "Right Anterior Vertical Occipital": "R aVOF",
    "Right Posterior Vertical Occipital": "R pVOF",
    "Right Early Visual": "R EV",
}

TRACT_ORDER = [
    # "Left Optic Radiation",
    "Left Arcuate",
    "Left Posterior Arcuate",
    "Left Temporo-parietal",
    "Left Anterior Vertical Occipital",
    "Left Posterior Vertical Occipital",
    "Left Early Visual",
    # "Right Optic Radiation",
    "Right Arcuate",
    "Right Posterior Arcuate",
    "Right Temporo-parietal",
    "Right Anterior Vertical Occipital",
    "Right Posterior Vertical Occipital",
    "Right Early Visual",
]

def _profiles(df, ses, metric, subjects, tracts):
    sub = df[(df["Pipeline"] == PIPELINE) & (df["Session"] == ses)]
    out = {}
    for t in tracts:
        M = (sub[sub["tractID"] == t]
             .pivot(index="Subject", columns="nodeID", values=metric)
             .reindex(index=subjects)
             .to_numpy())
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)
        sd[sd == 0] = 1.0
        out[t] = (M - mu) / sd
    return out


def run_population_similarity(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(df["Subject"].unique())
    tracts   = [t for t in TRACT_ORDER if t in df["tractID"].unique()]
    labels   = [TRACT_LABELS[t] for t in tracts]
    n        = len(tracts)

    records = []

    for metric in METRICS:
        Z1 = _profiles(df, SES1, metric, subjects, tracts)
        Z2 = _profiles(df, SES2, metric, subjects, tracts)

        corr_mat = np.full((n, n), np.nan)
        for i, j in tqdm(product(range(n), repeat=2), total=n * n):
            A, B = Z2[tracts[i]], Z1[tracts[j]]
            X = np.vstack([A, B])
            y = np.tile(np.arange(A.shape[0]), 2)
            D = pairwise_distances(X, metric="euclidean")
            corr_mat[i, j] = DiscrimOneSample(is_dist=True).statistic(D, y)

        for i, ti in enumerate(tracts):
            for j, tj in enumerate(tracts):
                records.append(dict(
                    metric    = metric,
                    tract_ses2= ti,
                    tract_ses1= tj,
                    disc  = corr_mat[i, j],
                    diagonal  = (i == j),
                ))

        _plot_heatmap(corr_mat, labels, metric, output_dir)

    results = pd.DataFrame(records)
    results.to_csv(output_dir / "population_similarity.csv", index=False)
    print(f"Results -> {output_dir / 'population_similarity.csv'}")

    _print_summary(results)
    return results


def _plot_heatmap(corr_mat, labels, metric, output_dir):
    n = len(labels)

    fig, ax = plt.subplots(figsize=((6/1.2), (6-1.2)/1.2))
    ax.set_title(
        f"HBN",
        fontsize=10, fontweight="bold",
    )

    sns.heatmap(
        corr_mat, ax=ax,
        xticklabels=labels,   # ses-1200
        yticklabels=labels,   # ses-Retest
        vmin=0.5, vmax=1, center=0.5,
        cmap="RdBu_r",
        annot=np.array([f"{v:.2f}".lstrip("0") for v in corr_mat.flatten()]).reshape(corr_mat.shape), fmt="", annot_kws={"size": 8},
        linewidths=0.4,
        cbar=False,
        # cbar_kws={"shrink": 0.5, "fraction": 0.03, "label": "Discriminability"},
        square=True,
    )

    ax.set_xlabel("First Split", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Second Split", fontsize=10, fontweight="bold", labelpad=8)
    ax.tick_params(labelsize=9)

    # Highlight diagonal (test-retest) with a black border per cell
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=False, edgecolor="black", lw=2,
            transform=ax.transData, clip_on=False,
        ))

    # White divider between left and right hemi blocks
    ax.axhline(6, color="white", lw=2.5)
    ax.axvline(6, color="white", lw=2.5)

    plt.tight_layout()
    path = output_dir / f"population_similarity_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap -> {path}")


def _print_summary(results):
    print(f"\n{'Metric':<8}  {'Comparison':<40}  {'mean D':>8}  {'n':>4}")
    print("-" * 65)

    for metric in METRICS:
        sub = results[results["metric"] == metric].copy()

        hemi_ses2 = sub["tract_ses2"].str.split(" ").str[0]
        hemi_ses1 = sub["tract_ses1"].str.split(" ").str[0]
        sub_ses2  = sub["tract_ses2"].str.replace("Left ", "").str.replace("Right ", "")
        sub_ses1  = sub["tract_ses1"].str.replace("Left ", "").str.replace("Right ", "")

        conditions = [
            ("test-retest (same bundle)",
             sub["diagonal"]),
            ("same hemi, diff sub-bundle",
             ~sub["diagonal"] & (hemi_ses2 == hemi_ses1)),
            ("diff hemi, same sub-bundle",
             ~sub["diagonal"] & (sub_ses2 == sub_ses1)),
            ("diff hemi, diff sub-bundle",
             ~sub["diagonal"] & (hemi_ses2 != hemi_ses1) & (sub_ses2 != sub_ses1)),
        ]

        for label, mask in conditions:
            vals = sub[mask]["disc"].dropna()
            if len(vals):
                print(f"{metric:<8}  {label:<40}  {vals.mean():>8.3f}  {len(vals):>4}")
        print()

VOF_SUBS = ["Posterior Vertical Occipital",
            "Anterior Vertical Occipital",
            "Early Visual"]


def run_subdivision_test(df, reps=1000, alt="neq"):
    subjects = sorted(df[df["Pipeline"] == PIPELINE]["Subject"].unique())
    tracts = [f"{h} {s}" for h in ("Left", "Right") for s in VOF_SUBS]

    records = []
    for metric in METRICS:
        Z1 = _profiles(df, SES1, metric, subjects, tracts)
        Z2 = _profiles(df, SES2, metric, subjects, tracts)

        p = {Z[t].shape[1] for Z in (Z1, Z2) for t in tracts}
        if len(p) != 1:
            raise ValueError(f"node counts differ across tracts: {p}")

        for shift in (1, 2):
            x1_rows, x2_rows, y = [], [], []
            item = 0
            for hemi in ("Left", "Right"):
                other = "Right" if hemi == "Left" else "Left"
                for si, s in enumerate(VOF_SUBS):
                    A  = Z2[f"{hemi} {s}"]
                    B1 = Z1[f"{hemi} {VOF_SUBS[(si + shift) % 3]}"]
                    B2 = Z1[f"{other} {s}"]

                    valid = ~(np.isnan(A).any(1) |
                              np.isnan(B1).any(1) |
                              np.isnan(B2).any(1))
                    for k in np.where(valid)[0]:
                        x1_rows += [A[k], B1[k]]
                        x2_rows += [A[k], B2[k]]
                        y += [item, item]
                        item += 1

            X1, X2, y = np.array(x1_rows), np.array(x2_rows), np.array(y)

            np.random.seed(42)
            d_cross, d_within, pval = DiscrimTwoSample().test(
                X2, X1, y, reps=reps, alt=alt, workers=1)

            records.append(dict(
                metric=metric, partner_shift=shift, n_items=item,
                d_diff_hemi_same_sub=d_cross,
                d_same_hemi_diff_sub=d_within,
                delta=d_cross - d_within,
                pval=pval,
            ))
            print(f"{metric}  shift={shift}  "
                  f"cross-hemi={d_cross:.3f}  within-hemi={d_within:.3f}  "
                  f"Δ={d_cross - d_within:+.3f}  p={pval:.4f}")


df = pd.read_csv("../compiled_hbn_results.csv")
run_population_similarity(df, Path("results_population"))
run_subdivision_test(df)
