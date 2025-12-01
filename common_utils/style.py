"""
Styling utilities, for reproducible, consistent, and easy-to-manage display.
"""
import logging

import matplotlib.pyplot as plt
import matplotlib as mpl

import cmocean

import seaborn as sns
import pandas as pd

from common_utils.colour_utils import gray15, gray80

'''>>> pandas display setting'''
# display all columns
pd.set_option(
    'display.max_columns',
    30, # int, None=unlimited
)
pd.set_option(
    'display.max_rows',
    30, # int, None=unlimited
)
'''<<< pandas display setting'''

'''>>> plt setting'''
#mpl.use('Qt5Agg') # for better wayland support
#mpl.use('MACOSX') # for mac
#mpl.use('Agg') # for ubuntu
# open plots in web browser to make it easier
mpl.use('WebAgg')
mpl.rcParams["webagg.open_in_browser"] = False
# bind only on loopback
mpl.rcParams["webagg.address"] = "127.0.0.1"
# pick a fixed port
mpl.rcParams["webagg.port"] = 8988

sns.set(
    #style="darkgrid", # gray background with grid
    #style="whitegrid", # white background with grid
    style="ticks", # "ticks" without background
    context="poster",
    #context="talk",
)

mpl.rcParams.update({
    #"font.family": ["Libertinus Sans"], # potentially for paper
    #"font.family": ["Latin Modern Sans"], # for thesis
    "font.family": ["LMSans12"], # on mac for thesis
    #"font.family": ["Libertinus Serif"], # for poster
    "figure.titlesize": 18,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.size": 12,
    "axes.titlepad": 8, # space between plot and ax title
    "axes.spines.top": False, # remove top & right spine
    "axes.spines.right": False,
    "pdf.compression": 9, # highly compress pdf
    "errorbar.capsize": 0.1,
    "scatter.edgecolors": "#cccccc", # gray80 (204, 204, 204)
    "lines.markeredgewidth": 0.5,
    "lines.markeredgecolor": "#cccccc", # gray80
})

# if to plot on dark background
"""
mpl.rcParams.update({
    "figure.facecolor": "#0f0f0f", # gray6 (15,15,15)
    "axes.facecolor": "#0f0f0f", # gray6
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#3d3d3d", # gray24 (61,61,61)
    "savefig.facecolor": "#0f0f0f", # gray6
})
"""

plt.tight_layout()
'''<<< plt setting'''

'''>>> logging setting'''
# Configure logging to include a timestamp with seconds
logging.basicConfig(
    level=logging.INFO,
    format='''\n%(asctime)s %(levelname)s: %(message)s\
            \n[in %(filename)s:%(lineno)d]''',
    datefmt='%Y%m%d %H:%M:%S',
)

#logging.info('This is an info message.')
#logging.warning('This is a warning message.')
#logging.error('This is an error message.')
'''<<< logging setting'''
