"""
contains utils that handles histology processing of mice.
"""

import os
import glob
import subprocess

from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib.pyplot as plt
import seaborn as sns

from pixels.error import PixelsError
from common_utils import file_utils, plot_utils, ml_utils
from common_utils.style import *
from common_utils.si_configs import *


def cluster_channels(rec, top_threshold, fig_dir):
    """
    Cluster top channels with k-means to determine the brain surface.

    params
    ===
    rec: spikeinterface recording obj, first five minute of ap band signal.

    top_threshold: int, depth along the probe for the top channels.

    return
    ===
    clustered_channels: pd dataframe, channel info with k-means clustering
        labels.
    """
    import spikeinterface as si

    # get channel locations
    chan_locs = rec.get_channel_locations()
    # get top channels only
    top_bool = chan_locs[:, 1] > top_threshold
    # get channel ids, then use select_channels
    chan_ids = rec.get_channel_ids()
    top_chans = chan_ids[top_bool]
    top_rec = rec.select_channels(top_chans)

    # separate shanks
    shanks = top_rec.split_by("group")
    logging.info(f"\n> getting traces from {len(shanks)} shanks")

    dfs = []
    for s, shank in shanks.items():
        # get channel traces
        logging.info(f"\n> getting traces from shank no.{s}")
        traces = shank.get_traces()

        logging.info(f"\n> clustering channels on shank {s} with k-means")
        Y_pred = ml_utils.k_means_clustering(
            data=traces.T,
            #n_clusters=2,
            fig_dir=fig_dir+f"_shank{s}",
        )
        # get channel locations
        shank_chan_locs = shank.get_channel_locations()
        df = pd.DataFrame(np.column_stack((shank_chan_locs, Y_pred)))
        df.columns = ["x", "y", "cluster"]

        # get noise level with MAD
        noise_levels = si.get_noise_levels(shank)
        df["noise_levels"] = noise_levels

        dfs.append(df)

    # concatenate shanks
    clustered_channels = pd.concat(
        dfs,
        axis=1,
        keys=shanks.keys(),
        names=["shank", "vars"],
    )

    return clustered_channels


def plot_clustered_channels(clustered_channels):
    """
    plot channel clusters after k-means.

    params
    ===
    clustered_channels: pd dataframe, channel info with k-means clustering
    labels.

    return
    ===
    clus_fig: matplotlib figure obj, figure of channel geometry & label.

    noise_fig: matplotlib figure obj, figure of channel depths vs noise, colour
        coded by label.
    """
    shanks = clustered_channels.columns.levels[0]

    if len(shanks) > 1:
        ncols = len(shanks)//2
        nrows = 2
    else:
        ncols = len(shanks)
        nrows = ncols

    noise_fig, noise_ax = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharey=True,
        sharex=True,
    )
    clus_fig, clus_ax = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharey=True,
        sharex=True,
    )

    for s, shank in enumerate(shanks):
        df = clustered_channels[shank]

        # flatten
        clus_ax_flat = clus_ax.flatten()
        noise_ax_flat = noise_ax.flatten()

        # plot channel clusters
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="cluster",
            ax=clus_ax_flat[s],
        )

        # plot channel clusters
        sns.scatterplot(
            data=df,
            x="y",
            y="noise_levels",
            hue="cluster",
            ax=noise_ax_flat[s],
        )

    return clus_fig, noise_fig


def find_rois(abbr: str):
    """
    Find regions of interest by their name.
    """
    rois:{
        "v1": "VISp",
        "hpf": ["ProS", "SUB", "CA", "DG", "FC", "IG"],
        "rsc": "RSP",
        "lp": "LP",
        "lgn": "LG",
    }


def get_hist_depths(hist_dir: str, mice: list, roi: list=None, dyes:list=["DiI"]):
    """
    get depth of each region from sharp-track histology results; depth zeros at the
    cortical surface.

    Histology file from Patri is named as `ExpMouseID`.

    params
    ===
    hist_dir: str, directory of the histology file

    mice: list, mouse IDs.

    cortex: list, names of cortical areas the probes enter.

    dyes: list, names of dyes the probes is painted with.

    returns
    ===
    (unit is millimeter)
    V1_borders, CA1_borders; np array, top and bottom of the region.
        for 'coarse' analysis, only the top & bottom of the whole region is required.
        For layer-specfic analysis, gather info from:

    V1_depths, CA1_depths; pandas DataFrame, depth info about all layers of the region.

    tip_depths: float, the deepest point of the probe.
    """

    # TODO jan 23 put this func inside of the main mice loop, i.e., use this
    # func at mouse level
    hist_dir = file_utils.make_path(hist_dir)  # convert str into path
    mice_hist = []

    tip_depths = np.zeros((len(mice), len(dyes)))
    # top & bottom border of the region
    V1_borders = np.zeros((len(mice), 2))
    HPF_borders = np.zeros((len(mice), 2))

    # use list append not vectorised array to deal with different segments
    V1_depths = []
    HPF_depths = []

    for m, mouse in enumerate(mice):
        # locate hist files for this mouse
        mouse_hist_dir = sorted(hist_dir.glob(f"*{mouse}*"))[0]

        if not mouse_hist_dir:  # make sure hist file exist
            raise PixelsError(f"have you processed histology of {name} yet?")

        # step 2: load .mat file, save it as nparray, and get histology depth info
        # for each region
        for d, dye in enumerate(dyes):
            logging.info(
                f"\n>>> getting depth information of {mouse} "
                f"from SHARP-Track results..."
            )
            depth_file = sorted(mouse_hist_dir.glob(f"*{dye}_depths*"))[0]
            depths = pd.read_csv(depth_file)  # get depths file

            tip_depths[m, d] = depths.iloc[:, 1].max()

            # V1
            logging.info(f"\n>>> getting V1 depth information of {mouse}...")
            V1 = depths.iloc[np.where(depths.loc[:, "acronym"].str.contains("VIS"))]
            V1_depths.append(V1)

            # min and max depths, i.e., borders of V1
            V1_borders[m] = np.array((V1.iloc[:, 0].min(), V1.iloc[:, 1].max()))

            # HPF
            hpf = ["ProS", "CA", "DG", "FC", "IG"]  # hippocampal formation
            logging.info(f"\n>>> getting HPF depth information of {mouse}...")
            HPFs = depths.iloc[
                np.where(
                    depths.loc[:, "acronym"].str.contains(r"\b{}.*".format("|".join(hpf)))
                )
            ]
            HPF_depths.append(HPFs)

            # min and max depths, i.e., borders of HPF
            HPF_borders[m] = np.array((HPFs.iloc[:, 0].min(), HPFs.iloc[:, 1].max()))

            # logging.info(f"\n For {mouse}, \n{V1_depths[m]} \n\n{HPF_depths[m]},\
            # \n and probe depth is {tip_depths[m]}mm.")


    return V1_depths, HPF_depths, V1_borders, HPF_borders, tip_depths



def get_regional_units(
    mice: list, sessions: dict, borders, tip_depths, **kwargs
) -> dict:
    """
    Get units from selected region of experiments.

    params
    ===
    sessions: dict, recording sessions of each mouse; mouse id is the key.

    mice: list, list of str of mouse ids.

    borders: np array, depth borders of the region zero at the cortical surface.
        For each region ofeeach mouse, borders array is arranged as [min_depth, max_depth].
        This arg defines the region (obviously).

    tip_depths: np array, depth of the probe from all mice, zeros at the cortical surface.

    return
    ===
    units: list, cluster ids of units from a selected region.
    """
    # kwargs.setdefault("name", "V1")
    units = {}

    for m, mouse in enumerate(mice):
        units[mouse] = []
        for i, ses in enumerate(sessions[mouse]):
            ses.set_probe_depth([tip_depths[m, i]])
            depth = borders[m]
            if depth.any():
                units[mouse].append(
                    ses.select_units(
                        min_depth=depth[0],
                        max_depth=depth[1],
                        **kwargs,
                    )
                )
            else:
                units.append([])

    return units


def get_depth_info(
    mice: list, num_selected_sessions: int, selected_mouse_sessions: dict
):
    """
    This function extract corrected depth info from saved .json file.

    params
    ===
    mice: list, list of str of mouse ids.

    num_selected_sessions: int, number of selected sessions in total.

    selected_mouse_sessions: dict, selected sessions for each mouse.

    return
    ===
    depths: nd array, summary of depth info from all info source.
        (mouse, session, (manipulator, histology, clustering)).

    df: pd dataframe, same info as depths, just for quick visual inspection.
    """
    depths = {}
    idx = []

    for m, mouse in enumerate(mice):
        for i, session in enumerate(selected_mouse_sessions[mouse]):
            # get name of the session
            name = session.name
            idx.append(name)

            depth_info = session.find_file(session.files[0]["depth_info"])
            if depth_info == None:
                raise PixelsError(f"have you corrected depth info of {name} yet?")

            # logging.info(f"\n> getting depth information of {name}")
            data = file_utils.load_json(depth_info)
            depths[name] = data

    df = pd.DataFrame(depths)

    return depths, df


def get_spike_width_boundary(exp: list, units: list):
    """
    This function gets the spike width boundary of interneuron and pyramidal neuron
    by using k-means to cluster units' median spike widths into 2 clusters, and
    taking the mean of the max and min as the boundary, which is the upper-limit of
    interneuron and lower-limit of pyramidal neuron.

    params
    ===
    exp: list, sessions from given mouse IDs.
    """
    spike_widths = exp.get_spike_widths(units=units)
    assert 0

    Y_pred = ml_utils.k_means_clustering(
        n_clusters=2,
        data=spike_widths,
    )
    assert 0
    # TODO



def get_V1_units(session):
    # get depth from saved borders file in processed
    # get borders, from updated borders
    borders_dir = session.processed / "borders.json"
    borders = json.load(open(borders_dir, mode="r"))

    # get probe depth
    depth_info_dir = session.processed / session.files[0]["depth_info"]
    # use clustering result as the final depth
    probe_depth = json.load(open(depth_info_dir, mode="r"))["clustering"]
    vf_border = borders["HPF"][0]
    hpf_lower = borders["HPF"][1]
    v1_top = borders["V1"][0]
    # V1 cortex top
    min_depth = probe_depth - v1_top
    # V1 cortex bottom
    max_depth = probe_depth - vf_border
    units = session.select_units(
        min_depth=min_depth,
        max_depth=max_depth,
        name="V1_good_units",
    )

    return units


def get_HPF_units(session):
    # get depth from saved borders file in processed
    # get borders, from updated borders
    borders_dir = session.processed / "borders.json"
    borders = json.load(open(borders_dir, mode="r"))

    # get probe depth
    depth_info_dir = session.processed / session.files[0]["depth_info"]
    # use clustering result as the final depth
    probe_depth = json.load(open(depth_info_dir, mode="r"))["clustering"]
    vf_border = borders["HPF"][0]
    hpf_lower = borders["HPF"][1]
    # V1 cortex top
    min_depth = probe_depth - vf_border
    # V1 cortex bottom
    max_depth = probe_depth - hpf_lower
    units = session.select_units(
        min_depth=min_depth,
        max_depth=max_depth,
        name="HPF_good_units",
    )

    return units


"""
def get_m2_interneurons(exp):
    return get_m2(exp, name_suffix="m2_interneurons", max_spike_width=m2_spike_width_boundary)

def get_str_pyramidals(exp):
    return get_str(exp, name="str_principal", min_spike_width=str_spike_width_boundary)

def get_str_interneurons(exp):
    return get_str(exp, name="str_interneurons", max_spike_width=str_spike_width_boundary)
"""
