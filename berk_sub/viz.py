import nibabel as nib
from AFQ.viz.fury_backend import visualize_volume, visualize_bundles
import AFQ.utils.streamlines as aus
from AFQ.viz.utils import COLOR_DICT
from AFQ.viz.utils import PanelFigure

from PIL import Image
import AFQ.utils.volume as auv
import numpy as np
from dipy.align import resample
from fury import window, actor
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from AFQ.viz.utils import COLOR_DICT
from tqdm import tqdm


from math import radians

rerun_tractogram=False
seg_sft_path = "sub-02_desc-bundles_tractography.trx"
fa_file = "sub-02_model-tensor_param-fa_dwimap.nii.gz"
t1_file = "sub-02_desc-masked_T1w.nii.gz"
volume_opacity_bundles=0.5
n_points_bundles=40

sbv_lims_bundles = [None, None]
t1_img = nib.load(t1_file)
shade_by_volume = nib.load(fa_file)
volume = nib.load(t1_file)

flip_axes = [False, False, False]

bundle_names = ["Right Posterior Arcuate", "Right Anterior Vertical Occipital", "Right Posterior Vertical Occipital", "Right Early Visual", "Right Optic Radiation"]


t1_img_in_dwi = resample(t1_img, shade_by_volume)
data = t1_img_in_dwi.get_fdata()
entire_density_map = np.zeros(
    (*t1_img_in_dwi.shape[:3], len(bundle_names))
)

slice_idx = int(data.shape[2] // 2 - 2)
r_bound = int(data.shape[0] // 2)
a_bound = int(data.shape[1] // 2 - 20)

slice_offsets = [-20, -10, 0, 10, 20]

if rerun_tractogram:
    seg_sft = aus.SegmentedSFT.fromfile(seg_sft_path, "same")

    seg_sft.bundle_names = bundle_names

    for offset in tqdm(slice_offsets):
        figure = visualize_volume(
            volume,
            opacity=1.0,
            flip_axes=flip_axes,
            interact=False,
            inline=False,
        )


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

        z = slice_idx + offset
        plane_mask = np.zeros(t1_img_in_dwi.shape[:3], dtype=np.uint8)
        # restrict to the same crop used for the 2D panels later, so the
        # plane you see here matches what actually gets shown downstream
        plane_mask[:r_bound, a_bound:, z:z+1] = 1
        plane_mask = resample(nib.Nifti1Image(plane_mask, shade_by_volume.affine), t1_img).get_fdata()

        plane_actor = actor.contour_from_roi(
            plane_mask,
            # affine=t1_img_in_dwi.affine,
            color=(0.8, 0, 0),
            opacity=0.8,
        )
        scene.add(plane_actor)

        show_m = window.ShowManager(
            scene=scene, size=(1200, 900),
            window_type="offscreen",
        )
        window.update_camera(show_m.screens[0].camera, None, scene)
        show_m.screens[0].controller.rotate((radians(90), 0), None)
        # show_m.screens[0].controller.rotate((0, radians(45)), None)

        show_m.render()
        show_m.window.draw()
        show_m.snapshot(f"tmp_init_snapshot_{offset}.png")

    for ii, bundle_name in enumerate(bundle_names):
        bundle_sl = seg_sft.get_bundle(bundle_name)
        bundle_density = auv.density_map(bundle_sl).get_fdata()
        entire_density_map[..., ii] = bundle_density 

    np.save("tmp_entire_density_map.npy", entire_density_map)
else:
    entire_density_map = np.load("tmp_entire_density_map.npy")

for offset in tqdm(slice_offsets):
    image_path = f"tmp_init_snapshot_{offset}.png"
    img = Image.open(image_path)

    rotated_img = img.rotate(-90, expand=True)

    width, height = rotated_img.size  # After rotation, this is 1125 x 1500
    new_width = width // 2            # 562 pixels
    new_height = height // 4          # 375 pixels

    left = width // 2
    top = 3 * height // 8 + 20
    right = 3 * width // 4
    bottom = 5 * height // 8

    cropped_img = rotated_img.crop((left, top, right, bottom))
    cropped_img = cropped_img.transpose(Image.FLIP_TOP_BOTTOM)
    cropped_img.save(f"tmp_cropped_shot_{offset}.png")

for jj in [-20, -10, 0, 10, 15, 20]:
    t1_slice = data[:r_bound, a_bound:, slice_idx+jj]
    t1_slice = np.flip(t1_slice, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(t1_slice.T, cmap="gray", origin="lower")

    for ii, name in enumerate(bundle_names):
        bundle_slice = entire_density_map[:r_bound, a_bound:, slice_idx+jj, ii]
        bundle_slice = np.flip(bundle_slice, axis=0)

        masked_bundle = np.ma.masked_where(bundle_slice == 0, bundle_slice)

        if np.any(bundle_slice > 0):
            rgb_color = np.array(COLOR_DICT[name])
            if np.max(rgb_color) > 1.0:
                rgb_color = rgb_color / 255.0
            if name == "Right Optic Radiation":
                max_opac = 0.5
            elif name == "Right Arcuate":
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
                vmax = np.max(masked_bundle)*0.5,  # saturate
                origin="lower",
            )

    ax.axis("off")

    plt.savefig("tmp_slice.png", dpi=300, bbox_inches="tight")

    img = Image.open("tmp_slice.png")

    width, height = img.size  # After rotation, this is 1125 x 1500

    left = 50
    top = 50
    right = 3 * width // 4
    bottom = 3 * height // 4

    cropped_img = img.crop((left, top, right, bottom))
    cropped_img = cropped_img.rotate(180)
    cropped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
    cropped_img.save(f"tmp_cropped_slice{jj}.png")

cfa_file = "sub-02_model-tensor_param-cfa_dwimap.nii.gz"
cfa_img = nib.load(cfa_file)
cfa_data = cfa_img.get_fdata()
cfa_resampled = np.zeros((*t1_img_in_dwi.shape[:3], 3))
for ii in range(3):
    chan_img = nib.Nifti1Image(cfa_data[..., ii], cfa_img.affine)
    cfa_resampled[..., ii] = resample(chan_img, t1_img_in_dwi).get_fdata()


for ii in [-20, -10, -5, 0, 10, 20]:
    cfa_slice = cfa_resampled[:r_bound, a_bound:, slice_idx + ii, :]
    cfa_slice = np.flip(cfa_slice, axis=0)
    v = cfa_slice.max(axis=-1)
    hi = np.percentile(v[v > 0], 99)
    cfa_slice = np.clip(cfa_slice / hi, 0, 1)

    fig_cfa, ax_cfa = plt.subplots(figsize=(8, 8))
    ax_cfa.imshow(np.transpose(cfa_slice, (1, 0, 2)), origin="lower")
    DENSITY_THRESH = 0.01

    for jj, name in enumerate(bundle_names):
        bundle_slice = entire_density_map[:r_bound, a_bound:, slice_idx + ii, jj]
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
    plt.savefig(f"tmp_cfa_slice{ii}.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cfa)

    img = Image.open(f"tmp_cfa_slice{ii}.png")
    width, height = img.size
    left = 50
    top = 50
    right = 3 * width // 4
    bottom = 3 * height // 4
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img = cropped_img.rotate(180)
    cropped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
    cropped_img.save(f"tmp_cropped_cfa_slice{ii}.png")

fig_leg, ax_leg = plt.subplots(figsize=(1.2, 6))
ax_leg.axis("off")

patches = []
for name in bundle_names:
    rgb = np.array(COLOR_DICT[name])
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    patches.append(mpatches.Patch(color=rgb, label=name))

ax_leg.legend(
    handles=patches,
    loc="center",
    frameon=False,
    fontsize=20,
    labelcolor="white",
    prop={"family": "Helvetica"},
    handlelength=1,
    handleheight=1.2,
    ncols=3,
    columnspacing=0.5,
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

#  1.5*8*7/6
pf = PanelFigure(10, 8, 11, 11, panel_label_kwargs=panel_label_kwargs)
pf.fig.patch.set_facecolor("black")

white_panel_label_kwargs = dict(
    color="white",
    bbox=dict(
        facecolor='none',
        edgecolor='none')
)

# pf.add_img("tmp_cropped_shot.png", slice(0, 2), slice(0, 3), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
subplot_label_pos=(0.3, 0.5)
pf.add_img("tmp_cropped_shot_-10.png", slice(0, 2), slice(0, 3), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_shot_0.png", slice(2, 4), slice(0, 3), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_shot_10.png", slice(4, 6), slice(0, 3), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_shot_20.png", slice(6, 8), slice(0, 3), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)

subplot_label_pos=(0.45, 0.3)
pf.add_img("tmp_cropped_slice-10.png", slice(0, 2), slice(3, 6), subplot_label_pos=subplot_label_pos)
subplot_label_pos=(0.6, 0.3)
pf.add_img("tmp_cropped_slice0.png", slice(2, 4), slice(3, 6), subplot_label_pos=subplot_label_pos)
pf.add_img("tmp_cropped_slice10.png", slice(4, 6), slice(3, 6), subplot_label_pos=subplot_label_pos)
subplot_label_pos=(0.4, 0.3)
pf.add_img("tmp_cropped_slice20.png", slice(6, 8), slice(3, 6), subplot_label_pos=subplot_label_pos)
subplot_label_pos=(0.4, 0.3)
pf.add_img("tmp_cropped_cfa_slice-10.png", slice(0, 2), slice(6, 9), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_cfa_slice0.png", slice(2, 4), slice(6, 9), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_cfa_slice10.png", slice(4, 6), slice(6, 9), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("tmp_cropped_cfa_slice20.png", slice(6, 8), slice(6, 9), subplot_label_pos=subplot_label_pos, panel_label_kwargs=white_panel_label_kwargs)

pf.add_img("tmp_legend.png", slice(0, 8), 9, add_panel_label=False)


# pf.add_img("endpoints_bar.png", slice(0, 4), slice(3, 6), subplot_label_pos=(0.3, 0.3))
# pf.add_img("results_population/population_similarity_dti_fa.png", slice(4, 8), slice(2, 6), subplot_label_pos=subplot_label_pos)
# pf.add_img("tmp_legend.png", slice(4, 8), slice(0, 2), add_panel_label=False, trim_buffer=60)
pf.format_and_save_figure("berk_extent.png")
