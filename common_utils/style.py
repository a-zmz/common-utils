"""
Styling utilities, for reproducible, consistent, and easy-to-manage display.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

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
    "font.family": ["Latin Modern Sans"], # for thesis
    #"font.family": ["Libertinus Serif"], # for poster
    #"xtick.labelsize": 16,
    #"ytick.labelsize": 16,
    #"axes.labelsize": 16,
    #"axes.titlesize": 18,
})

plt.tight_layout()
