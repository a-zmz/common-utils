"""
Some plotting/styling utilities. Split out from base so it can be used with behavioural
data plots too without the overhead of instantiating an Experiment.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import\
        LinearSegmentedColormap, ListedColormap, BoundaryNorm
import cmocean

import numpy as np

'''colour map'''
# NOTE to use registered cmap, simply do cmap='for_heatmap'
# NOTE for softer colours, set one RGB to 70, and modify others

purple = "#7d2fc7"
light_purple = "#be97e2"
light_gray = "#b7b7b7" # (183,183,183)
gray60 = "#999999" # (153,153,153)
gray48 = "#7a7a7a" # (122,122,122)
gray42 = "#6b6b6b" # (107,107,107)
darker_gray = "#4c4c4c" # (76,76,76)
gray24 = "#3d3d3d" # (61,61,61)
gray6 = "#0f0f0f" # (15,15,15)
super_dark_gray = "#0c0c0c" # (12,12,12)
black = "#000000" # (0,0,0)
water_blue = "#2f64c6"
dark_blue = "#072651"

# VR lick heatmap
blue_yellow = ["#072651", "#977713", "#ca9f1a", "#fdc721"]
# a more continuous colour map
cm_heatmap = LinearSegmentedColormap.from_list("cm_heatmap", blue_yellow)
bi_blue_yellow = ["#072651", "#fdc721"]
# a binary colour map, only 0 and 1
cm_bi_heatmap = ListedColormap(bi_blue_yellow)
bi_norm = BoundaryNorm([0, 0.5, 1], cm_bi_heatmap.N)

# VR behavour
# blues = ["#2f83c6", "#86a4ba"] # darker, lighter
#light_dark = ["#2f83c6", "#000000"] # blue, black

# purple, black & gray for light, dark & chance level
'''>>> vr behaviour colours'''
#light_dark = ["#7d2fc7", "#000000"] # purple, black
light_dark = [purple, super_dark_gray] # purple, super dark gray
lgt_orange = "#e2b28a" # light orange
data_chance = ["#86a4ba", "#999999"] # light blue, gray
# different light trial lengths, from short to long
light_lengths = [purple, "#8A43CC", "#9758D2", "#A46DD7", "#B182DD",
                 light_purple]
cm_light = LinearSegmentedColormap.from_list("cm_light", light_lengths)
dark_lengths = ["#02314d", "#1B455E", "#345A70", "#4D6E82", "#678394",
                "#8097A6"]
cm_dark = LinearSegmentedColormap.from_list("cm_dark", dark_lengths)
chance_lengths = [darker_gray, "#5B5B5B", gray42, gray48, "#898989",
                  gray60]
cm_chance = LinearSegmentedColormap.from_list("cm_chance", chance_lengths)
cm_lengths = {
    "light": cm_light,
    "dark": cm_dark,
    "chance": cm_chance,
}
'''vr behaviour colours<<<'''

# regional & neuronal types
'''>>> v1 oranges'''
#V1_oranges = ["#e8702a", "#ffbe4f", "#e39159", "#e69565", "#c6762f",
#"#c6962f", "#c6c22f"]
V1_orange = ["#c65f2f", "#c6782f", "#c6a02f"] # orange, yellowish orange, greenish orange
cm_v1 = LinearSegmentedColormap.from_list("cm_v1", V1_orange)
'''v1 oranges<<<'''

# TODO nov7 should i keep separating colour & plot utils or put them together?
# also see what is the best way to do colours: cmap or sns palette or just
# dictionary like `regions`

'''>>> hpf blues'''
#HPF_blues = ["#0ea7b5", "#6bd2db", "#2f83c6", "#59a6e3", "#65ade6",
#"#2f5fc6", "#2fc6be"]
HPF_blue = ["#2f64c6", "#2f87c6", "#2fc3c6"] # blue, greenish blue
cm_hpf = LinearSegmentedColormap.from_list("cm_v1", HPF_blue)
'''hpf blues<<<'''

# hue_order=colors.keys() will order seaborn legends accordingly
regions = {
    "V1": V1_orange[0],
    "HPF": HPF_blue[0],
    "na": gray60,
}

# cell type: regular-spiker1, regular-spiker2, fast-spiker
cell_types = ["rs1", "rs2", "fs"]
V1 = dict(zip(cell_types, V1_orange))
HPF = dict(zip(cell_types, HPF_blue))

regional_cell_type = {
    "V1": V1,
    "HPF": HPF,
    "Unidentified": gray60,
}

# >>> snake plot colours >>>
# colour hex `into the void`
snake_colours = ["#001931", "#00164f", "#033476", "#2b529b", "#5574af",
                 "#7f97c3", "#b80000"]
# clearer transition
#cm_snake = ListedColormap(snake_colours)
# softer transition
cm_snake = LinearSegmentedColormap.from_list("cm_snake", snake_colours)
# <<< snake plot colours <<<

# >>> z-score heatmap colours >>>
# clip top & bottom colour of cmocean thermal or matplotlib cubehelix
#base = plt.get_cmap("cmo.thermal") # better association with heat 
base = plt.get_cmap("cubehelix") # better discrimination around 0
clipped = base(np.linspace(0.1, 0.9, 256))
cm_zscore = LinearSegmentedColormap.from_list("cm_zscore", clipped, N=256)
# <<< z-score heatmap colours <<<

# NOTE july 29 2025:
# nice cmaps:
"""
cmap_names = ["cubehelix", "cmo.thermal_r_i", "cmo.oxy", "cmo.dense_i",
"cmo.deep_i", "cmo.tempo_r", "cividis"]
"""
