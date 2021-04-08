import os
import sys
import glob
import pandas as pd
import numpy as np
import h5py

from aa_code_utils import aa3toidx
from pdb_utils import parse_pdb


class Rotamer2AA(object):

    def __init__(self):
        self.mat = np.load("rotamer2aa_ukn.npy")

    def __call__(self, x):
        x = np.exp(x)
        x = x @ self.mat
        x = np.max(x.reshape(-1, 21, 34), axis=2)[:, 0:20]
        x = x / np.sum(x, axis=1, keepdims=True)
        return x


def pdb_summary(df_pdb, verbose=False):
    df_pdb_ca = df_pdb.query("atom_name == 'CA'").drop_duplicates(subset=["chain", "res_no"])
    chains = df_pdb_ca["chain"].drop_duplicates()
    if verbose:
        for chain in chains:
            df_chain = df_pdb_ca.query("chain == '{}'".format(chain))
            n1 = df_chain["res_no"].max() - df_chain["res_no"].min() + 1
            n2 = len(df_chain)
            print("chain {} seq len {} # CA {}".format(chain, n1, n2))
    return df_pdb_ca, chains


def analyze_pdb(csv_path, pdb_path, dist_cutoff=2.0):
    rot2aa = Rotamer2AA()
    df_pred = pd.read_csv(csv_path, index_col=0)
    xyz_pred = df_pred[["x", "y", "z"]].values
    df_pdb = parse_pdb(pdb_path)
    df_pdb_ca, chains = pdb_summary(df_pdb)
    for chain in chains:
        df_chain = df_pdb_ca.query("chain == '{}'".format(chain)).sort_values(by="res_no")
        n1 = df_chain["res_no"].max() - df_chain["res_no"].min() + 1
        n2 = len(df_chain)
        if n1 != n2:
            continue
        xyz_gt = df_chain[["x", "y", "z"]].values
        dists = np.linalg.norm(xyz_pred[np.newaxis, :, :] - xyz_gt[:, np.newaxis, :], axis=2)
        min_dists = np.min(dists, axis=1)
        pred_idxs = np.argmin(dists, axis=1)
        aa_pred_scores = df_pred.iloc[pred_idxs, 4:].values
        aa_pred_scores = rot2aa(aa_pred_scores)
        aa_pred = np.argmax(aa_pred_scores, axis=1)
        aa_gt = aa3toidx(df_chain["res_name"].values)
        n_aa_correct = np.sum(aa_pred == aa_gt)
        pred_idxs[min_dists > dist_cutoff] = -1
        n_found_ca = np.sum(pred_idxs != -1)
        n_ca = pred_idxs.size
        print("chain {} ({}): {} found, {} AA correct".format(chain, n_ca, n_found_ca, n_aa_correct))
        df_chain_pred = pd.DataFrame(dict(dist=min_dists, aa_gt=aa_gt))
        df_chain_pred["x_gt"] = xyz_gt[:, 0]
        df_chain_pred["y_gt"] = xyz_gt[:, 1]
        df_chain_pred["z_gt"] = xyz_gt[:, 2]
        df_chain_pred["x_pred"] = xyz_pred[pred_idxs, 0]
        df_chain_pred["y_pred"] = xyz_pred[pred_idxs, 1]
        df_chain_pred["z_pred"] = xyz_pred[pred_idxs, 2]
        aa_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                 "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        for j, aa in enumerate(aa_list):
            df_chain_pred[aa] = aa_pred_scores[:, j]
        data_export_path = csv_path.replace(".csv", "_{}.csv".format(chain))
        df_chain_pred.to_csv(data_export_path)


def analyze_all(csv_path, pdb_path, pdb_list_path):
    pdbs = [p.strip() for p in open(pdb_list_path).readlines()]
    for pdb in pdbs:
        print("==================== {} ====================".format(pdb))
        my_csv_path = os.path.join(csv_path, "{}.csv".format(pdb))
        my_pdb_path = os.path.join(pdb_path, "{}.pdb".format(pdb))
        analyze_pdb(my_csv_path, my_pdb_path)


if __name__ == "__main__":
    csv_path = sys.argv[1]
    pdb_path = sys.argv[2]
    pdb_list_path = sys.argv[3]
    analyze_all(csv_path, pdb_path, pdb_list_path)
