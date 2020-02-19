import os
import argparse
import numpy as np
import pandas as pd
from m2m_utils import list_files, parse_h5_attrs, parse_pdb_id


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("master_df_path", type=str, help="Where to find the master df")
    p.add_argument("dfs_dir_path", type=str, help="Where to find parsed df files")
    p.add_argument("--output_path", type=str, default="./", help="Where to put dataset df files")
    p.add_argument("--output_prefix", type=str, default="", help="Prefix for output files")
    p.add_argument("--error_log", type=str, default=None, help="Log h5 files that coulnd't be processed")
    return p.parse_args()


def main():
    args = parse_args()
    df_master = pd.read_hdf(args.master_df_path, "df")
    os.makedirs(args.output_path, exist_ok=True)
    df_files = list_files(args.dfs_dir_path)
    df_avai_pdbs = pd.DataFrame(df_files, index=[parse_pdb_id(f) for f in df_files], columns=["path"])
    df_master = df_master.merge(df_avai_pdbs, left_on="PDB", right_index=True, how="inner")
    df_rotamers = None
    for i in range(len(df_master)):
        pdb, resolution, subset = df_master.iloc[i][["PDB", "resolution", "subset"]]
        my_df = pd.read_hdf(df_avai_pdbs.at[pdb, "path"], "df")[["rot_label", "aa_label", "O", "C", "CA", "N"]]
        my_df = my_df.dropna(subset=["O", "C", "CA", "N"])
        my_df["PDB"] = pdb
        my_df["resolution"] = resolution
        my_df["subset"] = subset
        print(pdb, resolution, subset, "{} residues".format(len(my_df)))
        if df_rotamers is None:
            df_rotamers = my_df
        else:
            df_rotamers = pd.concat((df_rotamers, my_df), axis=0, ignore_index=True)
    print(df_rotamers)
    print("mean O  coor", np.mean(df_rotamers["O"].values))
    print("mean C  coor", np.mean(df_rotamers["C"].values))
    print("mean CA coor", np.mean(df_rotamers["CA"].values))
    print("mean N  coor", np.mean(df_rotamers["N"].values))
    for subset in ["val", "test", "train"]:
        df_subset = df_rotamers.query("subset == '{}'".format(subset)).reset_index(drop=True)
        dest_path = os.path.join(args.output_path, "{}{}.h5".format(args.output_prefix, subset))
        df_subset.to_hdf(dest_path, "df")
                    
    
if __name__ == "__main__":
    main()