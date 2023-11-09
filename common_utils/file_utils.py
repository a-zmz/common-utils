"""
This contains utility functions to handle files.
"""
import os
from pathlib import Path
import pickle
import shutil


def make_path(target_dir):
    """
    convert directory string into PosixPath object.
    """
    if isinstance(target_dir, str):
        path = Path(target_dir).expanduser()
    else:
        path = target_dir.expanduser()

    return path


def save_as_pickle(data, file_dir: str):
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
