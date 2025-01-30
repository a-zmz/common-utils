"""
Styling utilities, for reproducible, consistent, and easy-to-manage display.
"""

import matplotlib.pyplot as plt

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
#import matplotlib 
# use At5Agg for better wayland support
#matplotlib.use('Qt5Agg')
# plot display
sns.set_theme(
    style="darkgrid", # "ticks" without background
    context="talk",
    font_scale=0.7, # 1.2 0.7 0.3
)
#plt.tight_layout()
