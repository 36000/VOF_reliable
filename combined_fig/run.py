from AFQ.viz.utils import PanelFigure

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


pf = PanelFigure(12, 8, 12, 16, panel_label_kwargs=panel_label_kwargs)
pf.fig.patch.set_facecolor("black")

white_panel_label_kwargs = dict(
    color="white",
    bbox=dict(
        facecolor='none',
        edgecolor='none')
)

pf.add_img("../hcp_sub/tmp_cropped_shot.png", slice(0, 2), slice(0, 3), subplot_label_pos=(0.4, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("../hcp_sub/tmp_cropped_top_shot.png", slice(2, 6), slice(0, 3), subplot_label_pos=(0.5, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("../hcp_sub/tmp_cropped_slice.png", slice(0, 4), slice(3, 6), subplot_label_pos=(0.7, 0.5))
pf.add_img("../hcp_sub/tmp_cropped_cfa_slice.png", slice(4, 8), slice(3, 6), subplot_label_pos=(0.55, 0.5), panel_label_kwargs=white_panel_label_kwargs)

pf.add_img("../hbn_sub/tmp_cropped_shot.png", slice(0, 2), slice(6, 9), subplot_label_pos=(0.4, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("../hbn_sub/tmp_cropped_top_shot.png", slice(2, 6), slice(6, 9), subplot_label_pos=(0.5, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("../hbn_sub/tmp_reverse_cropped_shot.png", slice(6, 8), slice(6, 9), subplot_label_pos=(0.4, 0.5), panel_label_kwargs=white_panel_label_kwargs)
pf.add_img("../hbn_sub/tmp_cropped_slice.png", slice(0, 4), slice(9, 12), subplot_label_pos=(0.7, 0.5))
pf.add_img("../hbn_sub/tmp_cropped_cfa_slice0.png", slice(4, 8), slice(9, 12), subplot_label_pos=(0.55, 0.5), panel_label_kwargs=white_panel_label_kwargs)

pf.add_img("../hcp_sub/tmp_legend.png", slice(6, 8), slice(0, 3), add_panel_label=False, trim_buffer=60)

pf.format_and_save_figure("combined_extent.png")

