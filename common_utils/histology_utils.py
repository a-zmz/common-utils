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

from pixels.error import PixelsError
from common_utils import file_utils


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


def get_hist_depths(hist_dir: str, mice: list, roi: list, dyes=["DiI"]: list):
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

    tip_depths = np.zeros(len(mice) * len(dyes))
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
        print(f">>> getting depth information of {mouse} from SHARP-Track results...\n")
        depths = pd.read_csv(sorted(mouse_hist_dir.glob("*depths*"))[0])  # get depths file
        assert 0

        tip_depths[m] = depths.iloc[:, 1].max()

        # V1
        print(f">>> getting V1 depth information of {mouse}...\n")
        V1 = depths.iloc[np.where(depths.loc[:, "acronym"].str.contains("VIS"))]
        V1_depths.append(V1)

        # min and max depths, i.e., borders of V1
        V1_borders[m] = np.array((V1.iloc[:, 0].min(), V1.iloc[:, 1].max()))

        # HPF
        hpf = ["ProS", "CA", "DG", "FC", "IG"]  # hippocampal formation
        print(f">>> getting HPF depth information of {mouse}...\n")
        HPFs = depths.iloc[
            np.where(
                depths.loc[:, "acronym"].str.contains(r"\b{}.*".format("|".join(hpf)))
            )
        ]
        HPF_depths.append(HPFs)

        # min and max depths, i.e., borders of HPF
        HPF_borders[m] = np.array((HPFs.iloc[:, 0].min(), HPFs.iloc[:, 1].max()))

        # print(f"\n For {mouse}, \n{V1_depths[m]} \n\n{HPF_depths[m]},\
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
    depths = np.zeros((len(mice), num_selected_sessions // len(mice), 4))
    idx = []

    for m, mouse in enumerate(mice):
        for i, session in enumerate(selected_mouse_sessions[mouse]):
            # get name of the session
            name = session.name
            idx.append(name)

            depth_info = session.find_file(session.files[0]["depth_info"])
            if depth_info == None:
                raise PixelsError(f"have you corrected depth info of {name} yet?")

            # print(f"> getting depth information of {name}")
            data = pd.read_json(
                depth_info,
                typ="series",
            )
            depths[m, i, :] = data.values
            cols = data.index.values

    # TODO: add level to df, so the first level of the df is mouse id
    df = pd.DataFrame(
        data=depths.reshape((num_selected_sessions, 4)),
        columns=cols,
        index=idx,
    )

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
