import nibabel as nib
from AFQ.viz.utils import PanelFigure

from PIL import Image
import numpy as np
from dipy.align import resample
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

rerun_tractogram=False
fa_file = "../berk_sub/sub-02_model-tensor_param-fa_dwimap.nii.gz"
t1_file = "../berk_sub/sub-02_desc-masked_T1w.nii.gz"
volume_opacity_bundles=0.5
n_points_bundles=40

sbv_lims_bundles = [None, None]
t1_img = nib.load(t1_file)
shade_by_volume = nib.load(fa_file)
volume = nib.load(t1_file)

flip_axes = [False, False, False]
figure = None

t1_img_in_dwi = resample(t1_img, shade_by_volume)
data = t1_img_in_dwi.get_fdata()

slice_idx = 93
r_bound = int(data.shape[0] // 2 + 20)
a_bound = int(data.shape[1] // 2 - 40)

t1_slice = data[:r_bound, a_bound:, slice_idx]
t1_slice = np.flip(t1_slice, axis=0)

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(t1_slice.T, cmap="gray", origin="lower")

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
cropped_img.save(f"tmp_cropped_slice_intro.png")

cfa_file = "../berk_sub/sub-02_model-tensor_param-cfa_dwimap.nii.gz"
cfa_img = nib.load(cfa_file)
cfa_data = cfa_img.get_fdata()

cfa_slice = cfa_data[:, a_bound:, slice_idx, :]
cfa_slice = np.flip(cfa_slice, axis=0)

v = cfa_slice.max(axis=-1)
hi = np.percentile(v[v > 0], 99)
cfa_slice = cfa_slice / hi

fig_cfa, ax_cfa = plt.subplots(figsize=(8, 8))
ax_cfa.imshow(np.transpose(cfa_slice, (1, 0, 2)), origin="lower")
ax_cfa.axis("off")
plt.savefig(f"tmp_cfa_slice_intro.png", dpi=300, bbox_inches="tight")
plt.close(fig_cfa)

img = Image.open(f"tmp_cfa_slice_intro.png")
width, height = img.size
left = 50
top = 50
right = width - 50
bottom = 3 * height // 4
cropped_img = img.crop((left, top, right, bottom))
cropped_img = cropped_img.rotate(180)
cropped_img.save(f"tmp_cropped_cfa_slice_intro.png")

panel_label_kwargs = dict(
    fontfamily="Helvetica-Bold",
    fontsize="xx-large",
    color="black",
    fontweight='bold',
    verticalalignment="top",
    bbox=dict(
        facecolor='none',
        edgecolor='none'))


pf = PanelFigure(6, 6, 6*1.1, 6, panel_label_kwargs=panel_label_kwargs)
pf.fig.patch.set_facecolor("black")

white_panel_label_kwargs = dict(
    color="white",
    bbox=dict(
        facecolor='none',
        edgecolor='none')
)

pf.add_img("Wernicke_VOF.png", slice(0, 2), slice(0, 3), subplot_label_pos=(0.4, 0.4))
pf.add_img("tmp_cropped_cfa_slice_intro.png", slice(2, 6), slice(0, 3), subplot_label_pos=(0.3, 0.4), panel_label_kwargs=white_panel_label_kwargs)

pf.add_img("Takemura_VOF.png", slice(0, 4), slice(3, 6), subplot_label_pos=(0.5, 0.3), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("Yeatman_VOF.png", slice(4, 6), slice(3, 6), subplot_label_pos=(0.23, 0.3))

pf.format_and_save_figure("viz_intro.png")
