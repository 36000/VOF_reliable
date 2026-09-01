import nibabel as nib
from AFQ.viz.fury_backend import visualize_volume, visualize_bundles
import AFQ.utils.streamlines as aus
from AFQ.viz.utils import COLOR_DICT
from AFQ.viz.utils import PanelFigure

from PIL import Image
import AFQ.utils.volume as auv
import numpy as np
from dipy.align import resample
from fury import window
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from AFQ.viz.utils import COLOR_DICT


from math import radians


seg_sft_path = "sub-627549_desc-bundles_tractography.trx"
seg_sft = aus.SegmentedSFT.fromfile(seg_sft_path, "same")
fa_file = "sub-627549_model-tensor_param-fa_dwimap.nii.gz"
t1_file = "sub-627549_desc-masked_T1w.nii.gz"
volume_opacity_bundles=0.5
n_points_bundles=40

sbv_lims_bundles = [None, None]
t1_img = nib.load(t1_file)
shade_by_volume = nib.load(fa_file)
volume = nib.load(t1_file)

flip_axes = [True, False, False]
figure = None

figure = visualize_volume(
    volume,
    opacity=1.0,
    flip_axes=flip_axes,
    interact=False,
    inline=False,
    figure=figure,
)

seg_sft.bundle_names = [
    "Left Posterior Arcuate", "Left Anterior Vertical Occipital",
    "Left Posterior Vertical Occipital", "Left Early Visual", "Left Optic Radiation",
    "Right Posterior Arcuate", "Right Anterior Vertical Occipital",
    "Right Posterior Vertical Occipital", "Right Early Visual", "Right Optic Radiation",
]

scene = visualize_bundles(
    seg_sft,
    img=t1_img,
    shade_by_volume=shade_by_volume,
    sbv_lims=sbv_lims_bundles,
    n_points=n_points_bundles,
    flip_axes=flip_axes,
    interact=False,
    inline=False,
    figure=figure,
    opacity=0.8,
    line_width=0.1,
)


show_m = window.ShowManager(
    scene=scene, size=(1200, 900),
    window_type="offscreen",
)
window.update_camera(show_m.screens[0].camera, None, scene)
show_m.render()
show_m.window.draw()
show_m.snapshot("tmp_top_shot.png")

show_m.screens[0].controller.rotate((radians(90), 0), None)

show_m.render()
show_m.window.draw()
show_m.snapshot("tmp_init_snapshot.png")

show_m.screens[0].controller.rotate((radians(-180), 0), None)

show_m.render()
show_m.window.draw()
show_m.snapshot("tmp_reverse_snapshot.png")

image_path = "tmp_init_snapshot.png"
img = Image.open(image_path)

rotated_img = img.rotate(90, expand=True)

width, height = rotated_img.size  # After rotation, this is 1125 x 1500

left = width // 2
top = 3 * height // 8 + 40
right = 3 * width // 4
bottom = 5 * height // 8

cropped_img = rotated_img.crop((left, top, right, bottom))
cropped_img.save("tmp_cropped_shot.png")

image_path = "tmp_reverse_snapshot.png"
img = Image.open(image_path)

rotated_img = img.rotate(90, expand=True)

width, height = rotated_img.size  # After rotation, this is 1125 x 1500

left = width // 2
top = 3 * height // 8
right = 3 * width // 4
bottom = 5 * height // 8 - 40

cropped_img = rotated_img.crop((left, top, right, bottom))
cropped_img = cropped_img.transpose(Image.FLIP_TOP_BOTTOM)
cropped_img.save("tmp_reverse_cropped_shot.png")


image_path = "tmp_top_shot.png"
img = Image.open(image_path)

width, height = img.size  # After rotation, this is 1125 x 1500

left = 3 * width // 8 - 30
top = 4 * height // 8 - 30
right = 5 * width // 8 + 30
bottom = 6 * height // 8 - 20

cropped_img = img.crop((left, top, right, bottom))
cropped_img.save("tmp_cropped_top_shot.png")

t1_img_in_dwi = resample(t1_img, shade_by_volume)
data = t1_img_in_dwi.get_fdata()
entire_density_map = np.zeros(
    (*t1_img_in_dwi.shape[:3], len(seg_sft.bundle_names))
)
for ii, bundle_name in enumerate(seg_sft.bundle_names):
    bundle_sl = seg_sft.get_bundle(bundle_name)
    bundle_density = auv.density_map(bundle_sl).get_fdata()
    entire_density_map[..., ii] = bundle_density 

slice_idx = 60
a_bound = int(data.shape[1] // 2)

t1_slice = data[:, :a_bound, slice_idx]
t1_slice = np.flip(t1_slice, axis=0)

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(t1_slice.T, cmap="gray", origin="lower")

for ii, name in enumerate(seg_sft.bundle_names):
    bundle_slice = entire_density_map[:, :a_bound, slice_idx, ii]
    bundle_slice = np.flip(bundle_slice, axis=0)

    masked_bundle = np.ma.masked_where(bundle_slice == 0, bundle_slice)

    if np.any(bundle_slice > 0):
        rgb_color = np.array(COLOR_DICT[name])
        if np.max(rgb_color) > 1.0:
            rgb_color = rgb_color / 255.0
        if name == "Left Optic Radiation":
            max_opac = 0.5
        elif name == "Left Arcuate":
            max_opac = 0.5
        else:
            max_opac = 1.0

        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"custom_{name}",
            [(rgb_color[0], rgb_color[1], rgb_color[2], 0.1), 
             (rgb_color[0], rgb_color[1], rgb_color[2], max_opac)],
        )

        im = ax.imshow(
            masked_bundle.T,
            cmap=custom_cmap,
            origin="lower",
        )

ax.axis("off")

plt.savefig("tmp_slice.png", dpi=300, bbox_inches="tight")

img = Image.open("tmp_slice.png")

width, height = img.size  # After rotation, this is 1125 x 1500

left = width // 8
top = 50
right = 7 * width // 8
bottom = 3 * height // 4

cropped_img = img.crop((left, top, right, bottom))
cropped_img.save("tmp_cropped_slice.png")

cfa_file = "sub-627549_model-tensor_param-cfa_dwimap.nii.gz"
cfa_img = nib.load(cfa_file)
cfa_data = cfa_img.get_fdata()
cfa_resampled = np.zeros((*t1_img_in_dwi.shape[:3], 3))
for ii in range(3):
    chan_img = nib.Nifti1Image(cfa_data[..., ii], cfa_img.affine)
    cfa_resampled[..., ii] = resample(chan_img, t1_img_in_dwi).get_fdata()

if np.nanmax(cfa_resampled) > 1.0:
    cfa_resampled = cfa_resampled / 255.0
cfa_resampled = np.clip(np.nan_to_num(cfa_resampled), 0, 1)

cfa_slice = cfa_resampled[:, :a_bound, slice_idx, :]
cfa_slice = np.flip(cfa_slice, axis=0)
v = cfa_slice.max(axis=-1)
hi = np.percentile(v[v > 0], 99)
cfa_slice = np.clip(cfa_slice / hi, 0, 1)

fig_cfa, ax_cfa = plt.subplots(figsize=(8, 8))
ax_cfa.imshow(np.transpose(cfa_slice, (1, 0, 2)), origin="lower")
DENSITY_THRESH = 0.01

for jj, name in enumerate(seg_sft.bundle_names):
    bundle_slice = entire_density_map[:, :a_bound, slice_idx, jj]
    bundle_slice = bundle_slice / np.max(bundle_slice)
    bundle_slice = np.flip(bundle_slice, axis=0)

    rgb = np.array(COLOR_DICT[name])
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0

    ax_cfa.contour(
        bundle_slice.T,
        levels=[DENSITY_THRESH],
        colors=[rgb],
        linewidths=0.8,
        zorder=3 + jj,
    )
ax_cfa.axis("off")
plt.savefig(f"tmp_cfa_slice.png", dpi=300, bbox_inches="tight")
plt.close(fig_cfa)

img = Image.open(f"tmp_cfa_slice.png")
width, height = img.size

left = width // 8
top = 50
right = 7 * width // 8
bottom = 3 * height // 4

cropped_img = img.crop((left, top, right, bottom))
cropped_img.save(f"tmp_cropped_cfa_slice.png")

fig_leg, ax_leg = plt.subplots(figsize=(1.2, 6))
ax_leg.axis("off")

patches = []
for name in seg_sft.bundle_names:
    rgb = np.array(COLOR_DICT[name])
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    patches.append(mpatches.Patch(color=rgb, label=name))

ax_leg.legend(
    handles=patches,
    loc="center",
    frameon=False,
    fontsize=12,
    labelcolor="white",
    prop={"family": "Helvetica"},
    handlelength=1.5,
    handleheight=1.2,
    ncols=1,
    # columnspacing=1.0,
)

plt.savefig("tmp_legend.png", dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig_leg)

panel_label_kwargs = dict(
    fontfamily="Helvetica-Bold",
    fontsize="xx-large",
    color="black",
    fontweight='bold',
    verticalalignment="top",
    bbox=dict(
        facecolor='none',
        edgecolor='none'))
subplot_label_pos=(0.4, 0.5)


pf = PanelFigure(6, 8, 12, 8, panel_label_kwargs=panel_label_kwargs)
pf.fig.patch.set_facecolor("black")

white_panel_label_kwargs = dict(
    color="white",
    bbox=dict(
        facecolor='none',
        edgecolor='none')
)

pf.add_img("tmp_cropped_shot.png", slice(0, 2), slice(0, 3), subplot_label_pos=(0.4, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_top_shot.png", slice(2, 6), slice(0, 3), subplot_label_pos=(0.5, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_slice.png", slice(0, 4), slice(3, 6), subplot_label_pos=(0.7, 0.5))
pf.add_img("tmp_cropped_cfa_slice.png", slice(4, 8), slice(3, 6), subplot_label_pos=(0.55, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_legend.png", slice(6, 8), slice(0, 3), add_panel_label=False, trim_buffer=60)
pf.format_and_save_figure("hcp_extent.png")
