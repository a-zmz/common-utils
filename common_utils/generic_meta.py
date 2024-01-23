"""
generic meta setup for data analysis
"""

import time

'''
mice
'''
# tests
msc = [
    "HFR20",
]
tom = [
    "TF061",
]

# estrous
oes_cohort_1 = [
    "ESCN00",
    "ESCN02",
    "ESCN03",
]

# vision in darkness
vd_cohort_1 = [
    "WDAN00",
    "WDAN01",
]
vd_cohort_2 = [
    "WDAN03",
    "WDAN04",
]
vd_cohort_3 = [
    "WDAN07",
    "WDAN06",
    "WDAN05",
]

'''
directories
'''
# data
ardbeg = "~/rochefortlab/arthur/"
storage = "~/storage/data/"
processed = "~/processed/"
interim = "~/interim/"
server_w = "~/w/arthur/"
server_v = "~/v/arthur/"

# meta
oestrus_meta = "/home/amz/w/arthur/JSONs"

# histology
#hist_dir = "/home/amz/rochefortlab/arthur/histology/" # raw & processed
hist_dir = processed + "/histology/" # local processed

# figures
fig_dir = f"/home/amz/w/arthur/plots/behaviour/"
crap_fig_dir = f"/home/amz/Pictures/crappy_plots/"
# TODO jan 3 mice is not defined, how to preset prefix here?
#fig_prefix = f"{time.strftime('%Y%m%d')}_{'_'.join(mice)}_"

# testing
test_data = "/home/amz/running_data/npx"
test_interim = "/home/amz/running_data/npx/interim"

'''
misc
'''
session_date_fmt = "%Y%m%d" # datetime format yyyymmdd
