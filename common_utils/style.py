"""
Styling utilities, for reproducible, consistent, and easy-to-manage display.
"""
import logging

import matplotlib.pyplot as plt
import matplotlib as mpl

import cmocean

import seaborn as sns
import pandas as pd

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
    "font.size": 14,
    #"font.family": ["Libertinus Sans"], # potentially for paper
    #"font.family": ["Latin Modern Sans"], # for thesis
    "font.family": ["LMSans12"], # on mac for thesis
    #"font.family": ["Libertinus Serif"], # for poster
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titlepad": 15,
    "axes.spines.top": False, # remove top & right spine
    "axes.spines.right": False,
    "pdf.compression": 9, # highly compress pdf
    "errorbar.capsize": 0.1,
    "scatter.edgecolors": "#cccccc", # light gray
    "lines.markeredgewidth": 0.5,
    "lines.markeredgecolor": "#cccccc", # light gray
})

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
