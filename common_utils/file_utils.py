"""
This contains utility functions to handle files.
"""
import os
import operator
import json
import yaml
import tempfile
from pathlib import Path
import pickle
import shutil

import pandas as pd
import numpy as np

def make_path(target_dir):
    """
    convert directory string into PosixPath object.
    """
    if isinstance(target_dir, str):
        path = Path(target_dir).expanduser()
    else:
        path = target_dir.expanduser()

    return path


def save_as_pickle(file_dir: str, data):
    """
    Save data as pickle.
    """
    path = make_path(file_dir)

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"\n>> Pickle saved to {path}.")


def load_pickle(file_dir: str):
    """
    Load pickle.
    """
    path = make_path(file_dir)

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"\n>> Pickle loaded from {path}.")

    return data


def copy_files(source: str, dests: list, basename: str) -> None:
    """
    Copy all files with the same basename from source to destination.

    params
    ===
    source: str, directory of source files.

    dests: list of str, directory of destinations.

    basename: str, name of the file before extension.
    """
    source = make_path(source)

    # find all files with the same basename in source dir
    files = filter(lambda name: name.startswith(basename), os.listdir(source))

    # if dests not a list then make it one
    dests = dests if isinstance(dests, list) else [dests]

    # copy them to destination
    for file in files:
        for dest in dests:
            # make it a path
            dest = make_path(dest)
            # only copy if does not exist
            if not (dest / file).exists():
                shutil.copy2(source / file, dest)
                print(f"> Copied {file} to {dest}.")
            else:
                #print("already copied, next")
                continue

def load_json(path: str):
    file = json.load(open(path, mode="r"))

    return file


def read_json(path: str, names: list) -> None:
    path = make_path(path)
    for name in names:
        metadata = pd.read_json(path / f"{name}.json")
        print(f"\n{name}\n {metadata}")


def write_json(metadata: dict, file: str) -> None:
    with file.open(mode="w") as fd:  # opens file for writing
        json.dump(metadata, fd)
        print(f"Metadata saved in {file}")


def save_behaviour_json(meta_dir: str, mice: list) -> None:
    meta_dir = file_utils.make_path(meta_dir)
    for mouse in mice:
        meta = []
        file = meta_dir / f"{mouse}.json"
        # input date
        if input(f"\nInput meta from previous date? ") == "yes":
            date = input("date(yyyymmdd): ")
        else:
            date = datetime.now().strftime("%Y%m%d") # save date & time
            metadata = {
                "date": date, # save date & time
                "bodyweight (g)": input(f"Bodyweight of {mouse}: "),
                "note": input("Note: "),
            }

            meta.append(metadata)

            if file.exists():
                # get # old meta
                old_meta = json.load(open(file, mode="r"))
                # extend old meta
                old_meta.extend(meta)
                meta = old_meta
                print("Merged some old data.")
            else:
                meta = meta

            # save data to json
            try:
                write_json(meta, file)
            except FileNotFoundError:
                print("File does not exist.")
                write_json(meta, Path(tempfile.gettempdir()) / f"{mouse}.json")
            except Exception as e: # pylint: # disable=broad-except
                write_json(meta, Path(tempfile.gettempdir()) / f"{mouse}.json")
                print(f"Exception raised while saving: {type(e)}")
                print("Please report this.")
 

def sort_json_by_date(meta_dir: str, mice: list, output_dir: str = None) -> None:
    for mouse in mice: # load meta
        meta = json.load(open(meta_dir / f"{mouse}.json", "r"))
        # sort by date
        meta.sort(key=operator.itemgetter("date"))
        # save file
        if output_dir == None:
            output_dir = meta_dir
        write_json(meta, output_dir / f"{mouse}.json")


def read_hdf5(path, key="df"):
    """
    Read a dataframe from a h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the h5 file to read.

    Returns
    -------
    pandas.DataFrame : The dataframe stored within the hdf5 file under the name 'df'.

    """
    df = pd.read_hdf(
        path_or_buf=path,
        key=key,
        index_col=None,
    )
    return df


def write_hdf5(path, df, key="df", mode="w"):
    """
    Write a dataframe to an h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the h5 file to write to.

    df : pd.DataFrame
        Dataframe to save to h5.

    key : str
        identifier for the group in the store.
        Default: "df".

    mode : str
        mode to open file.
        Default: "w" write.
        Options:
            "a": append, if file does not exists it is created.
            "r+": similar to "a" but file must exists.
    """
    df.to_hdf(
        path_or_buf=path,
        key=key,
        mode=mode,
        format='table',
        index=False,
        complevel=9,
        complib="blosc:lz4hc",
    )
    
    print(f'\n>> HDF5 saved to {path}.')

    return

def init_memmap(path, dtype=np.float32, shape=None, overwrite=False, readonly=False):
    """
    Initiate memory-map.

    params
    ===
    path: str or Path instance, path to file.

    shape: int or sequence of ints, shape or desired.

    dtype: data-type.
        Default: np.float32, to save some space but retain at least 3-decimal
        precision.

    overwrite: bool, whether to overwrite existing file.

    readonly: bool, whether to only read existing file.

    return
    ===
    memory-map array.
    """
    # make path a Path instance
    path = make_path(path)

    if path.exists():
        if readonly:
            mode="r"
            overwrite = False
        elif overwrite:
            # if it already exists, but overwrite
            mode="w+"
            # make sure readonly is false
            readonly = False
        else:
            # allows reading and writing
            mode="r+"
    else:
        # if does not exist, create and overwrite
        mode="w+"

    mmap = np.memmap(
        filename=path,
        dtype=dtype,
        mode=mode,
        shape=shape,
    )

    return mmap


def save_yaml(path, data):

    yaml.dump(data, open(path, "w"), indent=4)

    print(f"\n>> yaml saved to {path}.")

    return None


def load_yaml(path):

    values = yaml.load(open(path, "r"), Loader=yaml.Loader)

    print(f"\n>> {path} loaded.")

    return values
