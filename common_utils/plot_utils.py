"""
General utility functions for plotting.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from common_utils.file_utils import make_path
from common_utils.style import *


class Subplots2D:
    """
    This will give you the `fig, axes` output from a 2D spread of subplots that fits
    your data, and use extra subplots to plot legend & label.

    For how to take advantage of these extras, see pixtools.across_action_plot

    attributes
    ===
    legend : an extra subplot that can be used to create a legend
    to_label : the bottom left subplot, for adding axis labels to
    axes_flat : the axes but in a flatted array for easier iteration

    """
    def __init__(self, data, *args, **kwargs):
        s = math.sqrt(len(data) + 1)
        # TODO nov 7 consider to set figsize (35, 15) here?
        fig, axes = plt.subplots(round(s), math.ceil(s), *args, **kwargs)

        self.fig = fig
        self.axes = axes

        if axes.ndim == 1:
            self.axes_flat = list(axes)
            self.to_label = axes[-1]
        else:
            self.axes_flat = [ax for dim in axes for ax in dim]
            self.to_label = axes[-1][0]

        self.legend = self.axes_flat[-1]

        # hide excess axes that fill the grid
        for i in range(len(data), len(self.axes_flat)):
            self.axes_flat[i].set_visible(False)


def save(path, fig=None, nosize=False, use_pdfpages=False):
    """
    Save a figure to the specified path. If a file extension is not part of the path
    name, it is saved as a PDF. The current figure is used, or a specified figure can be
    passed as fig=<figure>.
    """
    path = make_path(path)

    if not fig:
        fig = plt.gcf()

    if not nosize:
        fig.set_size_inches(10, 10)

    if not path.suffix:
        path = path.with_suffix(".svgz")

    suffix = path.suffix.lower().lstrip(".")
    save_kwargs = {
        "bbox_inches": "tight",
        "transparent": True,
    }

    if suffix == "pdf":
        save_kwargs["dpi"] = 300
    elif "svg" not in path.suffix:
        save_kwargs["dpi"] = 600

    if suffix == "pdf" and use_pdfpages:
        with PdfPages(path) as pdf:
            pdf.savefig(fig, **save_kwargs)
    else:
        fig.savefig(path, format=suffix, **save_kwargs)

    if len(path.parts) > 3:
        path = "/".join(path.parts[-3:])
    logging.info(f"Figure saved to: {path}")

    # close all open figures
    plt.close(fig)


# TODO nov 7 this essentially is the same as Subplots2D class...
# do some default setting of the figure & axes
def get_subplots(nrows=1, ncols=2, figsize=(35, 15), sharex=True, sharey=True,
                 *args, **kwargs):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        layout="constrained",
    )

    return fig, axes


def plot_QQ(data):
    """
    Make Quantile-Quantile plot to check normality of data distribution. 

    params
    ===
    data: np array like.

    return
    ===
    fig
    """
    fig, ax = plt.subplot(figsize=(6, 6))

    stats.probplot(data, dist="norm", plot=ax)

    return fig

def make_fixed_subplots(nrows, ncols,
                        subplot_w=10,   # inches
                        subplot_h=5,   # inches
                        wpad=2,        # horizontal padding between subplots (inches)
                        hpad=2,        # vertical padding
                        margin_lr=2,   # left+right margin
                        margin_tb=2,   # top+bottom margin
                        **kwargs):
    fig_w = ncols * subplot_w + (ncols - 1) * wpad + 2 * margin_lr
    fig_h = nrows * subplot_h + (nrows - 1) * hpad + 2 * margin_tb
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             **kwargs)
    # Optionally tweak spacing to match wpad/hpad exactly
    fig.subplots_adjust(
        left=margin_lr/fig_w,
        right=1-margin_lr/fig_w,
        bottom=margin_tb/fig_h,
        top=1-margin_tb/fig_h,
        wspace=wpad/subplot_w,
        hspace=hpad/subplot_h,
    )
    return fig, axes


def get_handles_n_labels(ax):
    # Collect handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Remove duplicates
    unique_handles_labels = dict(zip(labels, handles))
    labels = list(unique_handles_labels.keys())
    handles = list(unique_handles_labels.values())

    return handles, labels

