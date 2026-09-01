import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from AFQ.viz.utils import PanelFigure


TRACT_LABELS = {
    "Left Anterior Vertical Occipital": "laVOF / Right: raVOF",
    "Left Arcuate": "lAF / Right: rAF",
    "Left Posterior Vertical Occipital": "lpVOF / Right: rpVOF",
    "Left Posterior Arcuate": "lpAF / Right: rpAF",
    "Left Early Visual": "lEV / Right: rEV",
}

n_cols = 2
n_rows = 3
col_width = 4.0
row_height = 0.15

fig_leg, ax = plt.subplots(figsize=(col_width * n_cols, row_height * n_rows))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# need a renderer to measure text widths for mixed bold/normal placement
fig_leg.canvas.draw()
renderer = fig_leg.canvas.get_renderer()

fontsize = 8
fontfamily = "Helvetica"

col_x_starts = [0, 0.7]
row_y_centers = [1.0, 0.5, 0]


def draw_mixed_text(ax, x0, y0, segments, fontsize, fontfamily, renderer):
    x = x0
    for text, bold in segments:
        t = ax.text(
            x, y0, text,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            fontfamily=fontfamily,
            va="center", ha="left",
            color="black",
        )
        bbox = t.get_window_extent(renderer=renderer)
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        x += bbox_axes.x1 - bbox_axes.x0


for i, (full_name, acronym_str) in enumerate(TRACT_LABELS.items()):
    row = i // n_cols
    col = i % n_cols
    x_start = col_x_starts[col]
    y_pos = row_y_centers[row]

    left_acr, right_acr = acronym_str.split(" / Right: ")

    segments = [
        (f"{full_name}: ", False),
        (left_acr, True),
        (" / Right: ", False),
        (right_acr, True),
    ]
    draw_mixed_text(ax, x_start, y_pos, segments, fontsize, fontfamily, renderer)

plt.savefig("tmp_legend.png", dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig_leg)

vmin=0.5
vmax=1.0
fig, ax = plt.subplots(figsize=(4/6/1.5, 8*7/6/1.5))

def _discrim_cmap(vmin=0.5, vmax=1.0, center=0.5, base="RdBu_r", n=256):
    cmap = plt.get_cmap(base)
    vrange = max(vmax - center, center - vmin)
    norm = mpl.colors.Normalize(center - vrange, center + vrange)
    cmin, cmax = norm([vmin, vmax])
    return mpl.colors.ListedColormap(cmap(np.linspace(cmin, cmax, n)))


sm = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    cmap=_discrim_cmap(),
)
cb = fig.colorbar(sm, cax=ax, orientation="vertical")
cb.set_label("Discriminability Between Bundle FA Tract Profiles", fontsize=14, fontfamily="Helvetica")
cb.ax.tick_params(labelsize=13)

path = "tmp_cbar.png"
fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig)

panel_label_kwargs = dict(
    fontfamily="Helvetica-Bold",
    fontsize="xx-large",
    color="black",
    fontweight='bold',
    verticalalignment="top",
    bbox=dict(
        facecolor='none',
        edgecolor='none'))

pf = PanelFigure(6, 7, 8*7/6, 4, panel_label_kwargs=panel_label_kwargs)
pf.add_img("../hcp_sub/results_population/population_similarity_dti_fa.png", slice(0, 3), slice(0, 6), subplot_label_pos=(0.25, 1))
pf.add_img("../hbn_sub/results_population/population_similarity_dti_fa.png", slice(3, 6), slice(0, 6), subplot_label_pos=(0.25, 1))
# pf.add_img("tmp_legend.png", slice(0, 6), slice(6, 7), add_panel_label=False, reduct_count=0)
pf.add_img("tmp_cbar.png", slice(6, 7), slice(0, 6), add_panel_label=False)
pf.format_and_save_figure("uniqueness.png")

