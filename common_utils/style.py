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
# plot display
sns.set_context("paper")
plt.tight_layout()
sns.set(font_scale=1.2) # 0.7
sns.set_style("ticks")
