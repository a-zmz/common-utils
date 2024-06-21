"""
Some plotting/styling utilities. Split out from base so it can be used with behavioural
data plots too without the overhead of instantiating an Experiment.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages


'''colour map'''
# NOTE to use registered cmap, simply do cmap='for_heatmap'
# NOTE for softer colours, set one RGB to 70, and modify others

gray = "#999999" # gray

# VR lick heatmap
blue_yellow = ["#072651", "#977713", "#ca9f1a", "#fdc721"]
cm_heatmap = LinearSegmentedColormap.from_list("cm_heatmap", blue_yellow)

# VR behavour
# blues = ["#2f83c6", "#86a4ba"] # darker, lighter
#light_dark = ["#2f83c6", "#000000"] # blue, black

# purple, black & gray for light, dark & chance level
light_dark = ["#7d2fc7", "#000000"] # purple, black
lgt_orange = "#e2b28a" # light orange
data_chance = ["#86a4ba", "#999999"] # light blue, gray

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
    "na": gray,
}

# cell type: regular-spiker1, regular-spiker2, fast-spiker
cell_types = ["rs1", "rs2", "fs"]
V1 = dict(zip(cell_types, V1_orange))
HPF = dict(zip(cell_types, HPF_blue))

regional_cell_type = {
    "V1": V1,
    "HPF": HPF,
    "Unidentified": gray,
}
# <<< plt setting
