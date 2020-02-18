import os
import glob
import argparse
import pandas as pd
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
from m2m_utils import list_files, parse_h5_attrs, parse_pdb_id, load_h5_map, save_h5_map


def generate_masked_data(h5_file, sigma=10, verbose=True):
    data = load_h5_map(h5_file)
    mask = np.zeros_like(data, dtype=np.float)
    df = parse_h5_attrs(h5_file, posfix="_xyz")
    pdb_id = parse_pdb_id(h5_file)
    if verbose:
        print("{} shape:".format(pdb_id), data.shape, "{} residues".format(len(df)))
    atoms = [col for col in df.columns if col[-4:]=="_xyz"]
    for i in range(len(df)):
        coors = df.iloc[i][atoms].values
        for coor in coors:
            if np.isnan(coor).any():
                continue
            else:
                mask[int(coor[2]), int(coor[1]), int(coor[0])] = 1
    mask = gaussian_filter(mask, sigma=sigma)
    mask = mask / np.max(mask)
    print("mask max", np.max(mask))
    data = data * mask
    return data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path", type=str, help="Where to find map h5 files")
    p.add_argument("output_path", type=str, default="./", help="Where to put parsed h5 files")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    h5_files = list_files(args.h5_path)
    for h5_file in h5_files:
        pdb_id = parse_pdb_id(h5_file)
        print("{} => {}".format(h5_file, pdb_id))
        try:
            data = generate_masked_data(h5_file)
            save_h5_map(data, os.path.join(args.output_path, "{}.h5".format(pdb_id)))
        except:
            print("couldn't process {}".format(pdb_id))


if __name__ == "__main__":
    main()