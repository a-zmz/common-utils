"""
This contains utility functions to handle files.
"""
import os
from pathlib import Path
import pickle
import shutil


def make_path(str_dir: str):
    """
    convert directory string into PosixPath object.
    """
    path = Path(str_dir).expanduser()

    return path


def save_as_pickle(data, file_dir: str):
    """
    Save data as pickle.
    """
    with open(file_dir, "wb") as f:
        pickle.dump(data, f)

    print(f">> Pickle saved as {file_dir}.")


def load_pickle(file_dir: str):
    with open(file_dir, "rb") as f:
        data = pickle.load(f)

    print(f">> Pickle loaded from {file_dir}.")

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
    source = Path(source).expanduser()

    # find all files with the same basename in source dir
    files = filter(lambda name: name.startswith(basename), os.listdir(source))

    # copy them to destination
    for file in files:
        for dest in dests:
            # make it a path
            dest = Path(dest).expanduser()
            # only copy if does not exist
            if not (dest / file).exists():
                shutil.copy2(source / file, dest)
                print(f"> {file} copied to {dest}.")
            else:
                print("already copied, next")
                continue
