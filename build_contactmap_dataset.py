import os
import argparse
import numpy as np
import pandas as pd
import h5py
from m2m_utils import list_files, parse_h5_attrs, parse_pdb_id


def quality_check(df, verbose=False):
    chains = set(df["chain"])
    results = dict()
    for chain in chains:
        my_df = df.query("chain == '{}'".format(chain))
        my_df = my_df.dropna(subset=["CA"])
        n_res = len(my_df)
        n_range = my_df["no"].max()-my_df["no"].min()+1
        results[chain] = (n_res==n_range) and n_res>2
    return results


def get_ca_coors(df, chain):
    my_df = df.query("chain == '{}'".format(chain)).dropna().sort_values(by="no")
    try:
        ca_coors = np.concatenate(my_df["CA"].values, axis=0).reshape(-1, 3)
    except ValueError:
        print(my_df["CA"].dropna())
        print(my_df["CA"].values)
    return ca_coors
    
       
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("master_df_path", type=str, help="Where to find the master df")
    p.add_argument("dfs_dir_path", type=str, help="Where to find parsed df files")
    p.add_argument("--output_path", type=str, default="./", help="Where to put dataset df files")
    p.add_argument("--output_prefix", type=str, default="", help="Prefix for output files")
    p.add_argument("--error_log", type=str, default=None, help="Log h5 files that coulnd't be processed")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()


def main():
    args = parse_args()
    df_master = pd.read_hdf(args.master_df_path, "df").drop_duplicates(subset="PDB")
    os.makedirs(args.output_path, exist_ok=True)
    df_files = list_files(args.dfs_dir_path)
    df_avai_pdbs = pd.DataFrame(df_files, index=[parse_pdb_id(f) for f in df_files], columns=["path"])
    df_master = df_master.merge(df_avai_pdbs, left_on="PDB", right_index=True, how="inner")
    ca_ca_dists = []
    for subset in ["val", "test", "train"]:
        print("==================== {} ====================".format(subset))
        dest_path = os.path.join(args.output_path, "{}{}.h5".format(args.output_prefix, subset))
        df_master_subset = df_master.query("subset == '{}'".format(subset))
        if args.verbose:
            print(df_master_subset)
        f = h5py.File(dest_path, "w")
        for i in range(len(df_master_subset)):
            pdb = df_master_subset.iloc[i]["PDB"]
            my_df = pd.read_hdf(df_avai_pdbs.at[pdb, "path"], "df")[["chain", "no", "O", "C", "CA", "N"]]
            chains = quality_check(my_df)
            if args.verbose:
                print(pdb, chains)
            for chain in chains:
                if chains[chain]:
                    pass
                else:
                    print("{}/{} skipped".format(pdb, chain))
                    continue
                ca_coors = get_ca_coors(my_df, chain)
                ca_ca_dist = 0.25*np.mean(np.linalg.norm(ca_coors[0:-1,:]-ca_coors[1:,:], axis=1))
                ca_ca_dists.append(ca_ca_dist)
                print("{}/{} mean CA-CA distance: {:.2f} A".format(pdb, chain, ca_ca_dist))
                ca_coors_ds = f.create_dataset("CA/{}/{}".format(pdb, chain), ca_coors.shape, dtype=np.float)
                ca_coors_ds[()] = ca_coors
        f.close()
    ca_ca_dists = np.array(ca_ca_dists)
    print("CA-CA distance mean: {:.2f} A std: {:.2f} A".format(np.mean(ca_ca_dists), np.std(ca_ca_dists)))
                    
    
if __name__ == "__main__":
    main()
