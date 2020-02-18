import os
import glob
import argparse
import pandas as pd
import numpy as np
import h5py
import re


def list_files(dir_path):
    return glob.glob(os.path.join(dir_path, "*.h5"))


def load_h5_map(h5_path):
    with h5py.File(h5_path) as f:
        data = f["data"][()]
        return data
        

def save_h5_map(data, h5_path):
    with h5py.File(h5_path, "w") as f:
        dest = f.create_dataset("data", data.shape, dtype="f")
        dest[()] = data
        
        
def parse_h5_attrs(h5_path, posfix=""):
    # build a dict based on attrs in h5 file
    raw_data = {} 
    with h5py.File(h5_path) as f:
        for attr in f["data"].attrs:
            items = attr.split("_")
            if len(items) != 3:
                raw_data[attr] = f["data"].attrs[attr]
            else:
                raw_data[items[0], int(items[1]), items[2]] = f["data"].attrs[attr]
    # build df from the dict
    parsed_data = []
    for i, key in enumerate(raw_data["residues"]):
        chain = key[0].decode("utf-8")
        loc = int(key[1].decode("utf-8"))
        datum = {"chain": chain, "no": loc,
                 "rot_label": raw_data["rot_label"][i].decode("utf-8"),
                 "aa_label": raw_data["aa_label"][i].decode("utf-8"),
                 "centers": raw_data["centers"][i]}
        # iterate over atoms
        for i, atom in enumerate(raw_data[chain, loc, "names"]):
            datum[atom.decode("utf-8")+posfix] = raw_data[chain, loc, "coors"][i]
        parsed_data.append(datum)
    df = pd.DataFrame(parsed_data)
    return df


def parse_pdb_id(file_path):
    pat = re.compile("([\w]{4})\.h5")
    return pat.search(file_path, -7).group(1)