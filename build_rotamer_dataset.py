import os
import argparse
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
    os.makedirs(args.output_path, exist_ok=True)
    df_files = list_files(args.dfs_dir_path)
    for df_file in df_files:
        pdb_id = parse_pdb_id(df_file)
        try:
            df = pd.read_hdf(df_file, "df")
            print(pdb_id, len(df))
        except OSError:
            print("couldn't process {}".format(df_file))
                    

if __name__ == "__main__":
    main()