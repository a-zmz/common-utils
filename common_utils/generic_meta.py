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
    "WDAN05",
    "WDAN06",
    "WDAN07",
]
vd_cohort_4 = [
    "VDCN01",
    "VDCN02",
    "VDCN04",
]
vd_cohort_5 = [ # female
    "VDCN05",
    "VDCN09",
]

'''
directories
'''
# data
ardbeg = "~/rochefortlab/arthur/"
storage = "~/storage/data/"
processed = "~/processed/"
interim = "~/interim/"
running = "~/running_data/"
server_w = "~/w/arthur/"
server_v = "~/v/arthur/"
datastore = "~/datastore/arthur/"

# meta
oestrus_meta = "~/w/arthur/JSONs/"
# generic/standard meta file for vr
standard_vr_meta_dir = "~/interim/virmen_data/az_standard_meta.csv"

# histology
#hist_dir = "/home/amz/datastore/arthur/data/histology/" # raw & processed
vd_hist_dir = processed + "histology/vision-in-darkness/" # local processed

# figures
fig_dir = f"{datastore}plots/"
local_fig_dir = f"~/Pictures/plots/"
# TODO jan 3 mice is not defined, how to preset prefix here?
#fig_prefix = f"{time.strftime('%Y%m%d')}_{'_'.join(mice)}_"

# testing
test_data = "~/running_data/npx/"
test_interim = "~/running_data/npx/interim/"

'''
misc
'''
session_date_fmt = "%Y%m%d" # datetime format yyyymmdd
