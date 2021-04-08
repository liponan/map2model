import os
import glob
import time
import json
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from aa_code_utils import *
from utils import *
from deepmap_data import *
from deepmap_net import *


class AAAccuracy(object):

    def __init__(self, matrix_path, device=None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.mat = torch.from_numpy(np.load(matrix_path, allow_pickle=True)).float()
        self.mat = self.mat.to(self.device)

    def __call__(self, scores, targets, ignore_index=20):
        scores = torch.mm(scores.squeeze(), self.mat)
        scores, _ = torch.max(scores.view(-1, 21, 34), 2)
        _, idxs = torch.max(scores.view(-1, 21), 1)
        gt = targets  # .cpu().data.numpy()
        pr = idxs  # .cpu().data.numpy()
        acc = torch.sum(gt[gt != ignore_index].long() == pr[gt != ignore_index].long())
        # print("GT", targets.cpu().data.numpy())
        # print("PR", idxs.cpu().data.numpy())
        acc = acc.double() / scores.size(0)  # float(scores.size(0))
        return acc


class AlphaLoss(nn.Module):

    def __init__(
        self, ca_weight=20, cat_weight=20, o_weight=10, c_weight=10,
        n_weight=10, pos_weight_factor=1.0, c_ca=1.56, n_ca=1.51, o_ca=2.43,
        rot_label=True, device="cpu"
    ):
        super(AlphaLoss, self).__init__()
        self.bce = None
        self.mse = nn.MSELoss(reduction="mean")
        self.n_rot_labels = (20, 162)[rot_label]
        self.ce = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=self.n_rot_labels
        )
        self.ca_weight = ca_weight
        self.c_weight = c_weight
        self.n_weight = n_weight
        self.o_weight = o_weight
        self.cat_weight = cat_weight
        self.pos_weight_factor = pos_weight_factor
        self.anchors = (c_ca, n_ca, o_ca)
        self.sig = nn.Sigmoid()
        self.rot_label = rot_label
        self.device = device

    def forward(
            self,
            outputs,
            o_targets,
            c_targets,
            ca_targets,
            n_targets,
            rot_targets
        ):
        pos_weight = self.pos_weight_factor   \
                     * torch.sum(ca_targets[:, 0, :, :, :] < 1).float()\
                     / torch.sum(ca_targets[:, 0, :, :, :] > 0).float()
        pos_weight = torch.clamp(pos_weight, min=1.0, max=10000.0)
        # print("pos weight", float(pos_weight.data))
        pos_weight = pos_weight.to(self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        ca_conf_loss = self.bce(outputs[:, 0, :, :, :], ca_targets[:, 0, :, :, :])
        loss = ca_conf_loss
        mask = ca_targets[:, 0, :, :, :] > 0
        if torch.sum(mask.float()) > 0:
            # CA
            ca_z_loss = self.mse(
                self.sig(outputs[:, 1, :, :, :][mask]),
                ca_targets[:, 1, :, :, :][mask]
            )
            ca_y_loss = self.mse(
                self.sig(outputs[:, 2, :, :, :][mask]),
                ca_targets[:, 2, :, :, :][mask]
            )
            ca_x_loss = self.mse(
                self.sig(outputs[:, 3, :, :, :][mask]),
                ca_targets[:, 3, :, :, :][mask]
            )
            # C
            c_z = self.anchors[0] * torch.tanh(outputs[:, 4, :, :, :][mask])
            c_y = self.anchors[0] * torch.tanh(outputs[:, 5, :, :, :][mask])
            c_x = self.anchors[0] * torch.tanh(outputs[:, 6, :, :, :][mask])
            c_z_loss = self.mse(c_z, (c_targets[:, 0, :, :, :][mask]))
            c_y_loss = self.mse(c_y, (c_targets[:, 1, :, :, :][mask]))
            c_x_loss = self.mse(c_x, (c_targets[:, 2, :, :, :][mask]))
            # N
            n_z = self.anchors[1] * torch.tanh(outputs[:, 7, :, :, :][mask])
            n_y = self.anchors[1] * torch.tanh(outputs[:, 8, :, :, :][mask])
            n_x = self.anchors[1] * torch.tanh(outputs[:, 9, :, :, :][mask])
            n_z_loss = self.mse(n_z, (n_targets[:, 0, :, :, :][mask]))
            n_y_loss = self.mse(n_y, (n_targets[:, 1, :, :, :][mask]))
            n_x_loss = self.mse(n_x, (n_targets[:, 2, :, :, :][mask]))
            # O
            o_z = self.anchors[2] * torch.tanh(outputs[:, 10, :, :, :][mask])
            o_y = self.anchors[2] * torch.tanh(outputs[:, 11, :, :, :][mask])
            o_x = self.anchors[2] * torch.tanh(outputs[:, 12, :, :, :][mask])
            o_z_loss = self.mse(o_z, (o_targets[:, 0, :, :, :][mask]))
            o_y_loss = self.mse(o_y, (o_targets[:, 1, :, :, :][mask]))
            o_x_loss = self.mse(o_x, (o_targets[:, 2, :, :, :][mask]))
            if self.rot_label:
                n_rot_labels = 162
            else:
                n_rot_labels = 20
            mask_expand = mask[:, None, :, :, :].expand(
                mask.size(0),
                self.n_rot_labels,
                mask.size(1),
                mask.size(2),
                mask.size(3),
            )
            cat_loss = self.ce(
                torch.transpose(
                    outputs[:, 13:(13+self.n_rot_labels), :, :, :][mask_expand]\
                    .view(self.n_rot_labels, -1),
                    0,
                    1,
                ),
                rot_targets[mask[:, None, :, :, :]].view(-1)
            )
            ca_zyx_loss = ca_z_loss + ca_y_loss + ca_x_loss
            c_zyx_loss = c_z_loss + c_y_loss + c_x_loss
            n_zyx_loss = n_z_loss + n_y_loss + n_x_loss
            o_zyx_loss = o_z_loss + o_y_loss + o_x_loss
            loss += (
                self.ca_weight * ca_zyx_loss +
                self.c_weight * c_zyx_loss +
                self.n_weight * n_zyx_loss +
                self.o_weight * o_zyx_loss +
                self.cat_weight * cat_loss
            )
        return loss


def ca_zyx_error(outputs, targets, mask):
    sig = nn.Sigmoid()
    z = sig(outputs[:, 0, :, :, :][mask]) - targets[:, 0, :, :, :][mask]
    y = sig(outputs[:, 1, :, :, :][mask]) - targets[:, 1, :, :, :][mask]
    x = sig(outputs[:, 2, :, :, :][mask]) - targets[:, 2, :, :, :][mask]
    err = float(torch.mean(torch.norm(torch.cat((z.view(-1, 1), y.view(-1, 1), x.view(-1, 1)), dim=1), dim=1)))
    return err


def zyx_error(outputs, targets, mask, anchor):
    # dist = anchor * torch.exp(torch.tanh(outputs[:, 0, :, :, :][mask]))
    z = torch.tanh(outputs[:, 0, :, :, :][mask]).view(-1, 1)
    y = torch.tanh(outputs[:, 1, :, :, :][mask]).view(-1, 1)
    x = torch.tanh(outputs[:, 2, :, :, :][mask]).view(-1, 1)
    # delta = torch.norm(torch.cat((z, y, x), dim=1), dim=1)
    z = anchor * z - targets[:, 0, :, :, :][mask].view(-1, 1)
    y = anchor * y - targets[:, 1, :, :, :][mask].view(-1, 1)
    x = anchor * x - targets[:, 2, :, :, :][mask].view(-1, 1)
    err = float(torch.mean(torch.norm(torch.cat((z, y, x), dim=1), dim=1)))
    return err


class Metrics(object):

    def __init__(self, c_ca, n_ca, o_ca, cutoff=0.1, dx=0.25, downsample=8):
        self.anchors = [c_ca, n_ca, o_ca]
        self.cutoff = cutoff
        self.scale = downsample * dx

    def __call__(self, output, ca_targets, c_targets, n_targets, o_targets, rot_targets):
        sig = nn.Sigmoid()
        ca_conf = sig(output[:, 0, :, :, :])
        n_gt = int(torch.sum(ca_targets[:, 0, :, :, :]))
        positives = ca_conf > self.cutoff
        n_p = int(torch.sum(positives.float()))
        n_tp = torch.sum((ca_targets[:, 0, :, :, :])[positives])
        precision = float(n_tp / max(1, n_p))
        recall = float(n_tp / n_gt)
        mask = ca_targets[:, 0, :, :, :] > 0
        ca_err = self.scale * ca_zyx_error(output[:, 1:4, :, :, :], ca_targets[:, 1:4, :, :, :], mask)
        c_err = zyx_error(output[:, 4:7, :, :, :], c_targets, mask, self.anchors[0])
        n_err = zyx_error(output[:, 7:10, :, :, :], n_targets, mask, self.anchors[1])
        o_err = zyx_error(output[:, 10:13, :, :, :], o_targets, mask, self.anchors[2])
        rot_cat = 0
        if torch.sum(mask.float()) > 0:
            n_rot_labels = output.size(1) - 13
            mask_expand = mask[:, None, :, :, :].expand(
                mask.size(0),
                n_rot_labels,
                mask.size(1),
                mask.size(2),
                mask.size(3)
            )
            rot_cat = accuracy(
                torch.transpose(
                    output[:, 13:(13+n_rot_labels), :, :, :][mask_expand]\
                        .view(n_rot_labels, -1),
                    0,
                    1
                ),
                rot_targets[mask[:, None, :, :, :]].view(-1))
        return n_gt, n_p, precision, recall, ca_err, c_err, n_err, o_err, rot_cat


def extract(scores, cutoff=0.5, downsample=8, dx=0.25):
    ca_scores = nn.Sigmoid()(scores[0:4, :, :, :].data)
    mask = ca_scores[0, :, :, :] > cutoff
    # print("mask", mask.size())
    ca_conf = ca_scores[0, :, :, :][mask].view(-1, 1)
    # print("conf", conf.size())
    uvw = torch.nonzero(mask)
    # print("uwv", uvw.size())
    u = ca_scores[1, :, :, :][mask] + uvw[:, 0].float()
    v = ca_scores[2, :, :, :][mask] + uvw[:, 1].float()
    w = ca_scores[3, :, :, :][mask] + uvw[:, 2].float()
    ca_coors = downsample * dx * torch.cat(
        (u.view(-1, 1), v.view(-1, 1), w.view(-1, 1)),
        dim=1
    )
    # print("ca_coors", ca_coors.size())
    n_rot_labels = output.size(1) - 13
    cat = scores[13:, :, :, :][mask[None, :, :, :]\
        .expand(n_rot_labels, mask.size(0), mask.size(1), mask.size(2))]
    cat = torch.transpose(cat.view(n_rot_labels, -1), 0, 1)
    # print("cat", cat.size())
    return ca_conf, ca_coors, cat


def train(model, ds, params, device):
    model = model.to(device)
    loss_fn = AlphaLoss(
        ca_weight=params["loss"]["ca_weight"],
        c_weight=params["loss"]["c_weight"],
        n_weight=params["loss"]["n_weight"],
        o_weight=params["loss"]["o_weight"],
        cat_weight=params["loss"]["cat_weight"],
        pos_weight_factor=params["loss"]["pos_weight_factor"],
        c_ca=params["loss"]["c_ca"],
        n_ca=params["loss"]["n_ca"],
        o_ca=params["loss"]["o_ca"],
        rot_label=ds.rot_label,
        device=device
    )
    metric_fn = Metrics(
        c_ca=params["loss"]["c_ca"],
        n_ca=params["loss"]["n_ca"],
        o_ca=params["loss"]["o_ca"],
        cutoff=params["loss"]["cutoff"]
    )
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    shuffle = params["shuffle"]
    n_ep = params["n_epochs"]
    lr = params["lr"]
    backprop_every = params["backprop_every"]
    backup_every = params["backup_every"]
    backup_path = params["backup_path"]
    dl = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    seen = 0
    for i in range(n_ep):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for _j, (x, y_coor, y_rot) in enumerate(dl):
            if torch.sum(y_coor[:, 0, :, :, :]) < 100:
                continue
            x = x.to(device)
            y_coor = y_coor.to(device)
            y_rot = y_rot.to(device)
            model.train()
            output = model(x.float())
            seen += x.size(0)
            loss = loss_fn(
                outputs=output,
                ca_targets=y_coor[:, 0:4, :, :, :].float(),
                c_targets=y_coor[:, 4:7, :, :, :].float(),
                n_targets=y_coor[:, 7:10, :, :, :].float(),
                o_targets=y_coor[:, 10:13, :, :, :].float(),
                rot_targets=y_rot.long()
            )
            loss.backward()
            if seen % backprop_every == 0:
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                print("{:6d} loss {:8.4f}".format(seen, loss.item()))
                n_gt, n_p, pre, rec, ca_err, c_err, n_err, o_err, rot_cat = metric_fn(
                    output=output,
                    ca_targets=y_coor[:, 0:4, :, :, :].float(),
                    c_targets=y_coor[:, 4:7, :, :, :].float(),
                    n_targets=y_coor[:, 7:10, :, :, :].float(),
                    o_targets=y_coor[:, 10:13, :, :, :].float(),
                    rot_targets=y_rot.long()
                )
                print("       GT{:4d}  detected{:6d}  precision {:.3f}  recall {:.3f}  rot {:.3f}"
                      .format(n_gt, n_p, pre, rec, rot_cat))
                print("       CA {:.3f}  C {:.3f}  N {:.3f}  O {:.3f}".format(ca_err, c_err, n_err, o_err))
                if seen % backup_every == 0:
                    torch.save(model.state_dict(), os.path.join(backup_path, "latest.pt"))


def val(model, ds, params, device):
    model = model.to(device)
    # model.eval()
    model.train()
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    metric_fn = Metrics(
        c_ca=params["loss"]["c_ca"],
        n_ca=params["loss"]["n_ca"],
        o_ca=params["loss"]["o_ca"],
        cutoff=params["loss"]["cutoff"]
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    data = []
    with torch.no_grad():
        for _j, (x, y_coor, y_rot) in enumerate(dl):
            x = x.to(device)
            y_coor = y_coor.to(device)
            y_rot = y_rot.to(device)
            output = model(x.float())
            n_gt, n_p, pre, rec, ca_err, c_err, n_err, o_err, rot_cat = metric_fn(
                output=output,
                ca_targets=y_coor[:, 0:4, :, :, :].float(),
                c_targets=y_coor[:, 4:7, :, :, :].float(),
                n_targets=y_coor[:, 7:10, :, :, :].float(),
                o_targets=y_coor[:, 10:13, :, :, :].float(),
                rot_targets=y_rot.long()
            )
            print("       GT{:4d}  detected{:6d}  precision {:.3f}  recall {:.3f}  rot {:.3f}"
                  .format(n_gt, n_p, pre, rec, rot_cat))
            print("       CA {:.3f}  C {:.3f}  N {:.3f}  O {:.3f}".format(ca_err, c_err, n_err, o_err))
            output_np = output.data.cpu().numpy()
            input_np = x.data.cpu().numpy()
            label_np = y_coor.data.cpu().numpy()
            np.save(f"debug/val_x_{_j}.npy", input_np)
            np.save(f"debug/val_y_{_j}.npy", label_np)
            np.save(f"debug/val_z_{_j}.npy", output_np)
            data.append(dict(n_gt=n_gt, n_p=n_p, precision=pre, recall=rec,
                             rotamer_precision=rot_cat,
                             filename=f"debug/val_y_{_j}.npy"))
    df = pd.DataFrame(data)
    df.to_csv("debug/report.csv")


def predict(model, ds, params, device):
    model = model.to(device)
    # model.eval()
    model.train()
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    data = []
    with torch.no_grad():
        for _j, (x, y_coor, y_rot) in enumerate(dl):
            x = x.to(device)
            y_coor = y_coor.to(device)
            y_rot = y_rot.to(device)
            output = model(x.float())
            ca_conf, ca_coors, rot_cat = extract(output[0, :, :, :, :], cutoff=params["cutoff"])
            if ca_conf.size(0) == 0:
                print("[{:3d}] raw{:6d} nms{:6d}".format(_j, 0, 0))
            else:
                nms_idxs = nms(ca_conf, ca_coors, min_dist_cutoff=params["min_dist_cutoff"])
                print("[{:3d}] raw{:6d} nms{:6d}".format(_j, ca_conf.size(0), len(nms_idxs)))
                ca_conf = ca_conf[nms_idxs, :].data.cpu().numpy()
                ca_coors = ca_coors[nms_idxs, :].data.cpu().numpy()
                rot_cat = rot_cat[nms_idxs, :].data.cpu().numpy()
                data = np.concatenate((ca_conf, ca_coors, rot_cat), axis=1)
                df = pd.DataFrame(data)
                df.to_csv("predict/{}.csv".format(str(_j).zfill(9)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", type=str, default=None, help="train | validate | predict")
    p.add_argument("params", type=str, default=None, help="JSON file with parameters")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to a trained model")
    p.add_argument("--device", "-d", type=int, default=0, help="GPU device number")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()


def main():
    args = parse_args()
    params = json.load(open(args.params))
    seed = params["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AlphaNet(
        n_out=(33, 175)[params["rot_label"]],
        n_filters=params["net"]["n_filters"],
        bottleneck=params["net"]["bottleneck"],
        track_running_stats=params["net"]["track_running_stats"]
    )
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        print("Trained model {} loaded".format(args.model))
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device(f"cuda:{args.device}")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    if args.command == "train":
        print("Train DeepMap")
        pdb_list = params["train"]["pdb_list"]
        data_root = params["train"]["data_root"]
        ds_train = DeepMapDataset(
            pdb_list,
            data_root,
            rot_label=params["rot_label"],
            normalize=params["train"]["normalize"],
            transform=params["train"]["transform"]
        )
        print("Train set: {} boxes".format(len(ds_train)))
        train(model, ds_train, params["train"], device)
    elif args.command == "validate":
        print("Validate DeepMap")
        pdb_list = params["val"]["pdb_list"]
        data_root = params["val"]["data_root"]
        ds_val = DeepMapDataset(
            pdb_list,
            data_root,
            rot_label=params["rot_label"],
            normalize=True
        )
        print("Validation set: {} boxes".format(len(ds_val)))
        val(model, ds_val, params["val"], device)
    elif args.command == "predict":
        print("Predict with DeepMap")
        pdb_list = params["val"]["pdb_list"]
        data_root = params["val"]["data_root"]
        ds_pre = DeepMapDataset(
            pdb_list,
            data_root,
            rot_label=params["rot_label"],
            normalize=True,
            build_report=True
        )
        print("Prediction: {} boxes".format(len(ds_pre)))
        predict(model, ds_pre, params["predict"], device)


if __name__ == "__main__":
    main()
