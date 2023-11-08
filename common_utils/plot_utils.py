"""
General utility functions for plotting.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from common_utils.file_utils import make_path


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


def save(path, fig=None, nosize=False):
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
        path = path.with_suffix('.pdf')

    if path.suffix == '.pdf':
        with PdfPages(path) as pdf:
            pdf.savefig(figure=fig, bbox_inches='tight', dpi=300)

    else:
        fig.savefig(path, dpi=1200)

    if len(path.parts) > 3:
        path = "/".join(path.parts[-3:])
    print("Figure saved to: ", path)

    # close all open figures
    plt.close("all")


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
    )

    return fig, axes
