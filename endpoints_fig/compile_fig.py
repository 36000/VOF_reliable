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

pf = PanelFigure(5, 8, 8, 10, panel_label_kwargs=panel_label_kwargs)
pf.add_img("../output_left_berk.png", slice(0, 4), slice(0, 2), subplot_label_pos=(0.5, 0.3))
pf.add_img("../output_left_hcp.png", slice(0, 4), slice(2, 4), subplot_label_pos=(0.5, 0.3))
pf.add_img("../hcp_sub/endpoints_bar.png", slice(0, 8), 4, subplot_label_pos=(0.4, 0.2), reduct_count=0)

pf.add_img("../output_right_berk.png", slice(4, 8), slice(0, 2), add_panel_label=False)
pf.add_img("../output_right_hcp.png", slice(4, 8), slice(2, 4), add_panel_label=False)

pf.format_and_save_figure("endpoints.png")
