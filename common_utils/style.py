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
    "font.family": ["LMSans10"], # on mac for thesis
    #"font.family": ["Libertinus Serif"], # for poster
    # >>> font size >>>
    "figure.titlesize": 18, # font size
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.title_fontsize": 12,
    "legend.fontsize": 10,
    "font.size": 10,
    "axes.titlepad": 8, # space between plot and ax title
    # <<< font size <<<
    # >>> font weight >>>
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    # <<< font weight <<<
    # >>> spine >>>
    "axes.spines.top": False, # remove top & right spine
    "axes.spines.right": False,
    # <<< spine <<<
    # >>> colours >>>
    "text.color": gray15,
    "scatter.edgecolors": gray80,
    "lines.markeredgecolor": gray80,
    # <<< colours <<<
    # >>> ticks >>>
    "ytick.major.pad": 2.5, # distance between label to tick
    "ytick.minor.pad": 2.0,
    "xtick.major.pad": 2.5,
    "xtick.minor.pad": 2.0,
    "ytick.major.size": 4, # tick length
    "ytick.minor.size": 2.5,
    "xtick.major.size": 4,
    "xtick.minor.size": 2.5,
    # <<< ticks <<<
    # >>> misc >>>
    "lines.markeredgewidth": 0.5, # marker edge width
    "pdf.compression": 9, # highly compress pdf
    "errorbar.capsize": 0.1, # error bar cap width
    # <<< misc <<<
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
