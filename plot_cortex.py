import neuropythy as ny
import AFQ.utils.streamlines as aus
import numpy as np
import AFQ.recognition.utils as abu
import nibabel as nib
from AFQ.viz.utils import COLOR_DICT
from dipy.tracking.streamline import transform_streamlines

import k3d
from k3d.headless import k3d_remote
from pyvirtualdisplay import Display
from selenium import webdriver

import matplotlib.pyplot as plt
 
import boto3
from itertools import chain
 
session = boto3.Session(profile_name='hcp')
credentials = session.get_credentials().get_frozen_credentials()
 
ny.config['data_cache_root'] = "~/AFQ_data/npythy_cache/"
ny.config['hcp_credentials'] = (credentials.access_key, credentials.secret_key)

_display = Display(visible=False, size=(2048, 2048))
_display.start()


def get_display_driver(width=2048, height=2048):
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(width, height)
    return driver


def save_png_headless(plot, out_png, width=2048, height=2048):
    remote = k3d_remote(plot, get_display_driver(width, height), width=width, height=height)
    try:
        remote.sync(hold_until_refreshed=True)
        with open(out_png, 'wb') as f:
            f.write(remote.get_screenshot())
    finally:
        remote.close()

def sphere_view_camera(surf, mask, distance_factor=2.6, up_hint=(0, 0, 1)):
    xyz = np.asarray(surf.coordinates, dtype=float).T
    xyz = xyz - xyz.mean(axis=0)            # match plot_surface_k3d's centering

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("empty mask -- nothing to point the camera at")

    direction = xyz[mask].mean(axis=0)      # outward normal through the patch
    direction /= np.linalg.norm(direction)

    radius = np.linalg.norm(xyz, axis=1).max()
    position = direction * radius * distance_factor

    up = np.asarray(up_hint, dtype=float)
    up -= up.dot(direction) * direction     # up must be perpendicular to the view
    if np.linalg.norm(up) < 1e-6:           # patch sits at a pole; any perp works
        up = np.cross(direction, [1.0, 0.0, 0.0])
    up /= np.linalg.norm(up)

    return [*position, 0.0, 0.0, 0.0, *up] 
 
def curvature_underlay(surf, dark=0.35, light=0.65):
    curv = np.asarray(surf.prop('curvature'), dtype=float)
    grey = np.where(curv > 0, dark, light)          # positive curvature -> sulcus
    return np.repeat(grey[:, None], 3, axis=1)
 
 
def composite_over(rgba, background):
    a = np.clip(rgba[:, 3:4], 0, 1)
    rgb = np.clip(rgba[:, :3], 0, 1)
    return rgb * a + background * (1 - a)
 
 
def rgb_to_k3d(rgb):
    c = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint32)
    return (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
 
 
def plot_surface_k3d(surf, rgb, out_html, name=None):
    verts = np.asarray(surf.coordinates, dtype=np.float32).T
    verts = verts - verts.mean(axis=0)              # center the sphere on the origin
 
    faces = getattr(surf.tess, 'indexed_faces', None)
    if faces is None:
        faces = surf.tess.faces
    faces = np.asarray(faces, dtype=np.uint32).T
 
    plot = k3d.plot(
        grid_visible=False,
        camera_auto_fit=True,
        axes_helper=0,
        background_color=0xFFFFFF)
    plot += k3d.mesh(
        verts, faces,
        colors=rgb_to_k3d(rgb),
        side='double',
        flat_shading=False,
        name=name or 'cortex')
 
    with open(out_html, 'w') as f:
        f.write(plot.get_snapshot())
    return plot
 

def build_roi_labels(roi_data, hemi, sub, roi_subset=None):
    hkey = hemi                                   # json uses 'left'/'right'
    hlh  = {'left': 'lh', 'right': 'rh'}[hemi]
    n    = sub.hemis[hlh].vertex_count

    # sanity: the JSON's vertex count must match the loaded surface
    assert n == roi_data['n_verts'][hemi], (
        f"vertex mismatch: surface has {n}, json expects "
        f"{roi_data['n_verts'][hemi]} — wrong subject/surface?")

    rois = roi_data['rois']
    names = roi_subset if roi_subset is not None else sorted(rois.keys())

    vc_labels   = np.zeros(n, dtype=int)
    for i, name in enumerate(names, start=1):
        idcs = np.asarray(rois[name][hkey], dtype=int)
        vc_labels[idcs]  = i
    return vc_labels


study = "hcp" # berk or hcp

if study == "berk":
    ny.config['data_cache_root'] = "~/AFQ_data/npythy_cache/"
    ny.config['freesurfer_subject_paths'] = '.'

    mgz   = nib.load('MVauto/mri/orig.mgz')


    vox2ras = mgz.header.get_vox2ras()      # voxel -> surface (tkr) RAS
    print("vox2ras", vox2ras)
    vox2ras_tkr = mgz.header.get_vox2ras_tkr()          # voxel -> scanner RAS
    print("vox2ras_tkr", vox2ras_tkr)

    # comes from from-ACPC_to-anat_mode-image_xfm.mat
    affine = np.asarray([
        [  0.9993,   0.0291,   0.024,   -0.7571],
        [ -0.0338,   0.9737,   0.2252, -21.1506],
        [ -0.0168,  -0.2258,   0.974,  18.6321],
        [  0.,       0.,       0.,       1.    ]])
    
    sub = ny.freesurfer_subject("MVauto")
    seg_sft_path = "sub-02_desc-bundles_tractography.trx"
    seg_sft = aus.SegmentedSFT.fromfile(seg_sft_path, "same")

    vc_labels_l = ny.load((
        f"sub-118_hemi-L_visuallabels.mgz"
    ))    
    
    vc_labels_r = ny.load((
        f"sub-118_hemi-R_visuallabels.mgz"
    ))

else:
    subject = "627549"
    affine = np.eye(4)
    vox2ras = np.eye(4)
    vox2ras_tkr = np.eye(4)

    seg_sft_path = f"hcp_sub/sub-{subject}_desc-bundles_tractography.trx"
    seg_sft = aus.SegmentedSFT.fromfile(seg_sft_path, "same")

    sub = ny.hcp_subject(subject)

    vc_labels_l = ny.load((
        f"lh.{subject}.mgz"
    ))    
    
    vc_labels_r = ny.load((
        f"rh.{subject}.mgz"
    ))

 
b_names = [
    "Early Visual",
    "Posterior Vertical Occipital",
    "Anterior Vertical Occipital",
]
 
for hemi in ["Right", "Left"]:
    if hemi == "Left":
        vc_labels = vc_labels_l
    else:
        vc_labels = vc_labels_r
 
    hemi_id = hemi.lower()[0] + "h"
    cortex = sub.hemis[hemi_id]
 
    hemi_surf = cortex.surface()
    hemi_surf_white = cortex.surface("white")
    sphere_surf = cortex.surface('sphere')
 
    tracts_on_cortex_by_bundle = {}
    counts_on_cortex = np.zeros(hemi_surf.vertex_count, dtype=np.int32)
    for b_name in b_names:
        full_b_name = f"{hemi} {b_name}"
        print(f"\n    Bundle: {full_b_name}")

        sft = seg_sft.get_bundle(full_b_name)


        sls = transform_streamlines(sft.streamlines, np.linalg.inv(affine))
        sls = transform_streamlines(sls, vox2ras_tkr @ np.linalg.inv(vox2ras))
 
        fg_array = np.array(abu.resample_tg(sls, 20))
        coverage = np.zeros(hemi_surf.vertex_count, dtype=np.int32)
        first = fg_array[:, 0, :]
        last  = fg_array[:, -1, :]

        if False:  # whether to mark vertices within threshold, or snap to nearest
            endpoints = np.vstack([first, last])
            nbrs = hemi_surf_white.vertex_hash.query_ball_point(endpoints, r=2.0, workers=-1)
            flat = np.fromiter(chain.from_iterable(nbrs), dtype=np.int64)
            coverage = np.bincount(flat, minlength=hemi_surf.vertex_count).astype(np.int32)
        else:
            d, idx = hemi_surf_white.vertex_hash.query(np.vstack([first, last]), k=1)
            print("median endpoint-to-surface distance (mm):", np.median(d))
            np.add.at(coverage, idx, 1)
 
        tracts_on_cortex_by_bundle[b_name] = coverage
        counts_on_cortex += coverage
 
 
    unique_labels = sorted(set([l for l in vc_labels if l > 0]))
    n_labels = len(unique_labels)
    cmap = plt.get_cmap('tab10')
    label_to_color = {label: cmap(i / max(n_labels - 1, 1))[:3] + (1.0,) for i, label in enumerate(unique_labels)}
 
 
    total_bundle_rgba = np.zeros((len(vc_labels), 4))
    b_per_v = np.zeros((len(vc_labels)), dtype=np.int32)
    for ii, b_name in enumerate(b_names):
        # tract_mask = tracts_on_cortex_by_bundle[b_name] >= np.sum(tracts_on_cortex_by_bundle[b_name]) / 1e4
        # max_count = np.percentile(tracts_on_cortex_by_bundle[b_name][tract_mask > 0], 99)
        tract_mask = tracts_on_cortex_by_bundle[b_name] >= 2
        bundle_rgba = np.zeros((len(vc_labels), 4))
        if "Early" in b_name:
            color = (0.0, 0.0, 1.0)
        else:
            color = COLOR_DICT[hemi + " " + b_name]
        bundle_rgba[tract_mask > 0, :3] = color
        bundle_rgba[tract_mask > 0, 3] = 0.8
        b_per_v[tract_mask > 0] += 1
        total_bundle_rgba += bundle_rgba
 
    total_bundle_rgba[b_per_v != 0, :] /= b_per_v[b_per_v != 0][:, None]
    total_bundle_rgba = np.clip(total_bundle_rgba, 0, 1)
 
    for label in unique_labels:
        color = label_to_color[label]
        bp = vc_labels == label
        bp = bp.astype(float)
        addr = hemi_surf.isolines(bp, 0.5, yield_addresses=True)[0]
        verts = np.unique(addr['faces'].ravel())
        band = np.zeros(hemi_surf.vertex_count, dtype=bool)
        band[hemi_surf.tess.index(verts)] = True 
        total_bundle_rgba[band, :] = (1.0, 1.0, 1.0, 1.0)
 
 
    # k3d has no per-vertex alpha, so flatten the overlay onto the curvature
    # underlay ourselves before handing the colors off.
    vertex_rgb = composite_over(total_bundle_rgba, curvature_underlay(hemi_surf))
    plot = plot_surface_k3d(
        sphere_surf, vertex_rgb,
        f"output_{hemi.lower()}_{study}.html",
        name=f"{hemi} sphere")
    
    plot.camera_auto_fit = False
    if study == "berk":
        plot.camera = sphere_view_camera(sphere_surf, np.logical_or(vc_labels == 10, vc_labels == 4))
    else:
        plot.camera = sphere_view_camera(sphere_surf, np.logical_or(vc_labels == 10, vc_labels == 4))
    save_png_headless(plot, f"output_{hemi.lower()}_{study}.png")
