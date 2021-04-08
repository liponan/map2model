import os
import sys
import glob
import pandas as pd
import numpy as np
import h5py


def nms(df, min_dist=3.0, verbose=False):
    df = df.sort_values(by="conf").reset_index(drop=True)
    n1 = len(df)
    xyz_all = df[["x", "y", "z"]].values
    idxs = [0]
    for i in range(1, len(df)):
        dists = np.linalg.norm(
            xyz_all[[i], :] - xyz_all[idxs, :], axis=1
        )
        if np.min(dists) > min_dist:
            idxs.append(i)
    df = df.iloc[idxs]
    n2 = len(df)
    if verbose:
        print("nms: {} => {}".format(n1, n2))
    return df


def merge_chunks(data_path):
    os.makedirs(os.path.join(data_path, "merge"), exist_ok=True)
    csv_path = os.path.join(data_path, "summary.csv")
    df = pd.read_csv(csv_path, index_col=0)
    df_file = pd.DataFrame(
        glob.glob(os.path.join(data_path, "0*.csv")),
        columns=["csv_path"]
    )
    df = df.iloc[:len(df_file)].join(df_file["csv_path"])
    pdbs = df["pdb"].drop_duplicates().values
    for i, pdb in enumerate(pdbs):
        df_pdb = df.query("pdb == '{}'".format(pdb))
        print("Processing {} ({})".format(pdb, len(df_pdb)))
        df_coor = []
        for j in range(len(df_pdb)):
            csv_path, x_offset, y_offset, z_offset = df_pdb.iloc[j][["csv_path", "x_offset", "y_offset", "z_offset"]]
            df_box = pd.read_csv(csv_path, index_col=0)
            df_box.columns = ["conf", "z", "y", "x"] + ["rot_{}".format(j) for j in range(163)]
            df_box["x"] = df_box["x"] + x_offset
            df_box["y"] = df_box["y"] + y_offset
            df_box["z"] = df_box["z"] + z_offset
            df_coor.append(df_box)
        df_coor = pd.concat(df_coor, axis=0)
        df_coor = nms(df_coor, min_dist=3.0, verbose=True)
        df_coor.to_csv(
            os.path.join(data_path, "merge", "{}.csv".format(pdb))
        )


if __name__ == "__main__":
    data_path = sys.argv[1]
    merge_chunks(data_path)
