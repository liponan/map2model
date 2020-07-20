import os
from glob import glob
import time
import json
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net_utilts import *
from aa_code_utils import *
from utils import *
import argparse


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


class AlphaNet(nn.Module):

    def __init__(self, n_in=1, n_out=179, n_filters=16, n_blocks=None, bottleneck=False):
        super(AlphaNet, self).__init__()
        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResBlock
        self.seen = 0
        if n_blocks is None:
            n_blocks = [3, 4, 6]
        self.in_layer = conv7x7(n_in, n_filters, stride=2, bias=False, batchnorm=True, activate=True)
        if bottleneck:
            self.out_layer = conv1x1(16 * n_filters, n_out, bias=True, batchnorm=False, activate=False)
            self.resnet = nn.Sequential(
                self.in_layer,
                block(n1=n_filters, n2=n_filters, stride=1),
                repeat_layers(block(n1=4 * n_filters, n2=n_filters, stride=1), n_blocks[0]-1),
                block(n1=4*n_filters, n2=2 * n_filters, stride=2),
                repeat_layers(block(n1=8 * n_filters, n2=2 * n_filters, stride=1), n_blocks[1] - 1),
                block(n1=8 * n_filters, n2=4 * n_filters, stride=2),
                repeat_layers(block(n1=16 * n_filters, n2=4 * n_filters, stride=1), n_blocks[2] - 1),
                self.out_layer
            )
        else:
            self.out_layer = conv1x1(4 * n_filters, n_out, bias=True, batchnorm=False, activate=False)
            self.resnet = nn.Sequential(
                self.in_layer,
                repeat_layers(block(n_in=n_filters, n_out=n_filters, stride=1), n_blocks[0]),
                block(n_in=n_filters, n_out=2 * n_filters, stride=2),
                repeat_layers(block(n_in=2 * n_filters, n_out=2 * n_filters, stride=1), n_blocks[1] - 1),
                block(n_in=2 * n_filters, n_out=4 * n_filters, stride=2),
                repeat_layers(block(n_in=4 * n_filters, n_out=4 * n_filters, stride=1), n_blocks[2] - 1),
                self.out_layer
            )

    def forward(self, x):
        x = self.resnet(x)
        return x


def rotate_data(x, axis1, axis2):
    rand_num = np.random.randint(0, 3)
    if rand_num == 1:
        x = x.transpose(axis1, axis2).flip(axis1)
    elif rand_num == 2:
        x = x.transpose(axis1, axis2).flip(axis2)
    elif rand_num == 3:
        x = x.flip(axis2).flip(axis1)
    else:
        pass
    return x


def data_trasform(x):
    x = rotate_data(x, 2, 3)
    x = rotate_data(x, 1, 3)
    x = rotate_data(x, 1, 2)
    return x


def data_noramlization(x):
    # x = x - torch.mean(x)
    # x = x / torch.max(torch.tensor(0.001), torch.std(x))
    x[x < 0] = 0
    x = x - np.mean(x)
    x = x / max(0.001, np.std(x))
    return x


def make_vector_map(coors, data_shape, downsample=1):
    output = np.zeros((4, *[int(np.ceil(i / downsample)) for i in data_shape]))
    for coor in coors:
        i = int(coor[2] / downsample)
        j = int(coor[1] / downsample)
        k = int(coor[0] / downsample)
        output[0, i, j, k] = 1
        output[1, i, j, k] = np.fmod(coor[2] / downsample, 1)
        output[2, i, j, k] = np.fmod(coor[1] / downsample, 1)
        output[3, i, j, k] = np.fmod(coor[0] / downsample, 1)
    return output


def make_field_map(coors, ca_coors, data_shape, downsample=1):
    output = np.zeros((3, *[int(np.ceil(i / downsample)) for i in data_shape]))
    for (coor, ca_coor) in zip(coors, ca_coors):
        i = int(ca_coor[2] / downsample)
        j = int(ca_coor[1] / downsample)
        k = int(ca_coor[0] / downsample)
        # dis = np.linalg.norm(coor - ca_coor)
        vec = (coor - ca_coor)
        output[0, i, j, k] = coor[2] / downsample - i
        output[1, i, j, k] = coor[1] / downsample - j
        output[2, i, j, k] = coor[0] / downsample - k
    return output


def make_cat_map(labels, ca_coors, data_shape, downsample=1):
    output = np.zeros((1, *[int(np.ceil(i / downsample)) for i in data_shape]))
    for (label, ca_coor) in zip(labels, ca_coors):
        i = int(ca_coor[2] / downsample)
        j = int(ca_coor[1] / downsample)
        k = int(ca_coor[0] / downsample)
        output[0, i, j, k] = label
    return output


class AlphaDataset(Dataset):

    def __init__(self, dataset_df_path, normalize=False, transform=False, downsample=1, n=-1, max_size=-1,
                 max_n_targets=1000, query=None):
        self.files = None
        self.df = pd.read_hdf(dataset_df_path, "df")
        if query is not None:
            self.df = self.df.query(query)
        if max_size > 0:
            self.df = self.df.query("i <= {0} and j <= {0} and k <= {0}".format(max_size))
        if 0 < n <= len(self.df):
            self.df = self.df.sample(n)
        self.df = self.df.reset_index(drop=True)
        self.n = len(self.df)
        self.normalize = normalize
        self.transform = transform
        self.downsample = downsample
        self.max_n_targets = max_n_targets

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        attr_path, mask_path = self.df.iloc[idx][["attr_path", "mask_path"]]
        # print("attr_path", attr_path)
        # print("mask_path", mask_path)
        with h5py.File(mask_path, "r") as f:
            data = f["data"][()]
        if self.normalize:
            data = data_noramlization(data)
        if self.transform:
            data = data_trasform(data)
        df_attr = pd.read_hdf(attr_path, "df")
        atom_coors = df_attr[["O", "C", "CA", "N"]].dropna(how="any", axis=0).values
        o_coors = np.concatenate(atom_coors[:, 0]).reshape(-1, 3)
        c_coors = np.concatenate(atom_coors[:, 1]).reshape(-1, 3)
        ca_coors = np.concatenate(atom_coors[:, 2]).reshape(-1, 3)
        n_coors = np.concatenate(atom_coors[:, 3]).reshape(-1, 3)
        o_field = make_field_map(o_coors, ca_coors, data.shape, downsample=self.downsample)
        c_field = make_field_map(c_coors, ca_coors, data.shape, downsample=self.downsample)
        ca_field = make_vector_map(ca_coors, data.shape, downsample=self.downsample)
        n_field = make_field_map(n_coors, ca_coors, data.shape, downsample=self.downsample)
        r_tokens = rot_to_idx(list(df_attr["rot_label"].values))
        r_labels = make_cat_map(r_tokens, ca_coors, data.shape, downsample=self.downsample)
        a_tokens = aa3toidx(list(df_attr["aa_label"].values))
        a_labels = make_cat_map(a_tokens, ca_coors, data.shape, downsample=self.downsample)
        data = data[np.newaxis, :, :, :]
        return data, (o_field, c_field, ca_field, n_field, r_labels, a_labels)


class AlphaLoss(nn.Module):

    def __init__(self, ca_zyx_weight=10, cat_weight=10, o_weight=5, c_weight=5, n_weight=5, pos_weight=100,
                 o_ca=2.50, c_ca=2.00, n_ca=2.00, device=None):
        super(AlphaLoss, self).__init__()
        self.bce = None
        self.mse = nn.MSELoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=162)
        self.ca_zyx_weight = ca_zyx_weight
        self.cat_weight = cat_weight
        self.o_weight = o_weight
        self.c_weight = c_weight
        self.n_weight = n_weight
        self.anchors = (o_ca, c_ca, n_ca)
        self.sig = nn.Sigmoid()

    def forward(self, outputs, o_targets, c_targets, ca_targets, n_targets, rot_targets):
        pos_weight = torch.sum(ca_targets[:, 0, :, :, :] < 1) / torch.sum(ca_targets[:, 0, :, :, :] > 0)
        # print("pos weight", float(pos_weight.data))
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        ca_conf_loss = self.bce(outputs[:, 0, :, :, :], ca_targets[:, 0, :, :, :])
        loss = ca_conf_loss
        mask = (outputs[:, 0, :, :, :] * ca_targets[:, 0, :, :, :]) > 0.1
        if torch.sum(mask.float()) > 0:
            ca_z_loss = self.mse(self.sig(outputs[:, 1, :, :, :][mask]), (ca_targets[:, 1, :, :, :][mask]))
            ca_y_loss = self.mse(self.sig(outputs[:, 2, :, :, :][mask]), (ca_targets[:, 2, :, :, :][mask]))
            ca_x_loss = self.mse(self.sig(outputs[:, 3, :, :, :][mask]), (ca_targets[:, 3, :, :, :][mask]))
            o_l = self.anchors[0] * torch.exp(torch.tanh(outputs[:, 4, :, :, :][mask]))
            o_z = self.anchors[0] * torch.tanh(outputs[:, 5, :, :, :][mask])
            o_y = self.anchors[0] * torch.tanh(outputs[:, 6, :, :, :][mask])
            o_x = self.anchors[0] * torch.tanh(outputs[:, 7, :, :, :][mask])
            # o_d = torch.sqrt(o_z * o_z + o_y * o_y + o_x * o_x)
            # o_z = o_l / o_d * o_z
            # o_y = o_l / o_d * o_y
            # o_x = o_l / o_d * o_x
            o_z_loss = self.mse(o_z, (o_targets[:, 0, :, :, :][mask]))
            o_y_loss = self.mse(o_y, (o_targets[:, 1, :, :, :][mask]))
            o_x_loss = self.mse(o_x, (o_targets[:, 2, :, :, :][mask]))
            c_l = self.anchors[1] * torch.exp(torch.tanh(outputs[:, 8, :, :, :][mask]))
            c_z = self.anchors[1] * torch.tanh(outputs[:, 9, :, :, :][mask])
            c_y = self.anchors[1] * torch.tanh(outputs[:, 10, :, :, :][mask])
            c_x = self.anchors[1] * torch.tanh(outputs[:, 11, :, :, :][mask])
            # c_d = torch.sqrt(c_z * c_z + c_y * c_y + c_x * c_x)
            # c_z = c_l / c_d * c_z
            # c_y = c_l / c_d * c_y
            # c_x = c_l / c_d * c_x
            c_z_loss = self.mse(c_z, (c_targets[:, 0, :, :, :][mask]))
            c_y_loss = self.mse(c_y, (c_targets[:, 1, :, :, :][mask]))
            c_x_loss = self.mse(c_x, (c_targets[:, 2, :, :, :][mask]))
            n_l = self.anchors[2] * torch.exp(torch.tanh(outputs[:, 12, :, :, :][mask]))
            n_z = self.anchors[2] * torch.tanh(outputs[:, 13, :, :, :][mask])
            n_y = self.anchors[2] * torch.tanh(outputs[:, 14, :, :, :][mask])
            n_x = self.anchors[2] * torch.tanh(outputs[:, 15, :, :, :][mask])
            # n_d = torch.sqrt(n_z * n_z + n_y * n_y + n_x * n_x)
            # n_z = n_l / n_d * n_z
            # n_y = n_l / n_d * n_y
            # n_x = n_l / n_d * n_x
            n_z_loss = self.mse(n_z, (n_targets[:, 0, :, :, :][mask]))
            n_y_loss = self.mse(n_y, (n_targets[:, 1, :, :, :][mask]))
            n_x_loss = self.mse(n_x, (n_targets[:, 2, :, :, :][mask]))
            mask_expand = mask[:, None, :, :, :].expand(mask.size(0), 163, mask.size(1), mask.size(2), mask.size(3))
            cat_loss = self.ce(
                torch.transpose(outputs[:, 16:179, :, :, :][mask_expand].view(163, -1), 0, 1),
                rot_targets[mask[:, None, :, :, :]].view(-1))
            ca_zxy_loss = ca_z_loss + ca_y_loss + ca_x_loss
            o_zxy_loss = o_z_loss + o_y_loss + o_x_loss
            c_zxy_loss = c_z_loss + c_y_loss + c_x_loss
            n_zxy_loss = n_z_loss + n_y_loss + n_x_loss
            loss += self.ca_zyx_weight * ca_zxy_loss
            loss += self.o_weight * o_zxy_loss + self.c_weight * c_zxy_loss + self.n_weight * n_zxy_loss
            loss += self.cat_weight * cat_loss
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
    z = torch.tanh(outputs[:, 1, :, :, :][mask]).view(-1, 1)
    y = torch.tanh(outputs[:, 2, :, :, :][mask]).view(-1, 1)
    x = torch.tanh(outputs[:, 3, :, :, :][mask]).view(-1, 1)
    # delta = torch.norm(torch.cat((z, y, x), dim=1), dim=1)
    z = anchor * z - targets[:, 0, :, :, :][mask].view(-1, 1)
    y = anchor * y - targets[:, 1, :, :, :][mask].view(-1, 1)
    x = anchor * x - targets[:, 2, :, :, :][mask].view(-1, 1)
    err = float(torch.mean(torch.norm(torch.cat((z, y, x), dim=1), dim=1)))
    return err


def metrics(outputs, o_targets, c_targets, ca_targets, n_targets, rot_targets, aa_targets, cutoff=0.1,
            o_ca=2.50, c_ca=2.00, n_ca=2.00, aaacc=None):
    sig = nn.Sigmoid()
    ca_conf = sig(outputs[:, 0, :, :, :])
    n_gt = int(torch.sum(ca_targets[:, 0, :, :, :]))
    positives = ca_conf > cutoff
    n_p = int(torch.sum(positives.float()))
    n_tp = torch.sum((ca_targets[:, 0, :, :, :])[positives])
    precision = n_tp / max(1, n_p)
    recall = n_tp / n_gt
    precision = float(precision)
    recall = float(recall)
    mask = (ca_conf * ca_targets[:, 0, :, :, :]) > cutoff
    ca_err = ca_zyx_error(outputs[:, 1:4, :, :, :], ca_targets[:, 1:4, :, :, :], mask)
    o_err = zyx_error(outputs[:, 4:8, :, :, :], o_targets, mask, o_ca)
    c_err = zyx_error(outputs[:, 8:12, :, :, :], c_targets, mask, c_ca)
    n_err = zyx_error(outputs[:, 12:16, :, :, :], n_targets, mask, n_ca)
    rot_cat = 0
    aa_cat = 0
    if torch.sum(mask.float()) > 0:
        mask_expand = mask[:, None, :, :, :].expand(mask.size(0), 163, mask.size(1), mask.size(2), mask.size(3))
        rot_cat = accuracy(torch.transpose(outputs[:, 16:179, :, :, :][mask_expand].view(163, -1), 0, 1),
                           rot_targets[mask[:, None, :, :, :]].view(-1))
        if aaacc is not None:
            aa_cat = aaacc(torch.transpose(outputs[:, 16:179, :, :, :][mask_expand].view(163, -1), 0, 1),
                           aa_targets[mask[:, None, :, :, :]].view(-1))
    return n_gt, n_p, precision, recall, o_err, c_err, ca_err, n_err, rot_cat, aa_cat


def train(model, ds, params, device, val_ds=None):
    n_iters = int(len(ds) / params["batch_size"])
    dataloader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"],
                            drop_last=True)
    model = model.to(device)
    model.train()
    aaacc = AAAccuracy(matrix_path="rotamer2aa_ukn.npy", device=device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = AlphaLoss()
    log = list()
    acc_loss = 0
    for i in range(params["epochs"]):
        t1 = time.time()
        for j, (x, y) in enumerate(dataloader):
            # print("x", x.size())
            # print("y[2]", y[2].size())
            x = x.to(device)
            o_field = y[0].to(device)
            c_field = y[1].to(device)
            ca_field = y[2].to(device)
            n_field = y[3].to(device)
            rot_labels = y[4].to(device)
            aa_labels = y[5].to(device)
            model.train()
            output = model(x)
            # print("output", output.size())
            loss = loss_fn(output, o_field.float(), c_field.float(), ca_field.float(), n_field.float(),
                           rot_labels.long())
            loss.backward()
            model.seen += x.size(0)
            if model.seen % params["back_every"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                acc_loss += x.size(0) * loss.data.cpu().item()
                if model.seen % params["print_every"] == 0:
                    acc_loss = acc_loss / float(params["print_every"])
                    eta = (n_iters - (j - 1)) / (j + 1) * (time.time() - t1)
                    print("seen {:6d}  loss: {:6.4f}  {:5.0f}s to go for this epoch"
                          .format(model.seen, acc_loss, eta))
                    n_gt, n_p, pre, rec, o_err, c_err, ca_err, n_err, rot_cat, aa_cat\
                        = metrics(output, o_field.float(), c_field.float(), ca_field.float(), n_field.float(),
                                  rot_labels.long(), aa_labels.long(), aaacc=aaacc)
                    print("    GT {:3d}  detected {:3d}  precision {:.3f}  recall {:.3f}  rot {:.3f}  aa {:.3f}"
                          .format(n_gt, n_p, pre, rec, rot_cat, aa_cat))
                    print("    CA {:.3f}  O {:.3f}  C {:.3f}  N {:.3f}".format(ca_err, o_err, c_err, n_err))
                    log.append(dict(seen=model.seen, loss=acc_loss))
                    acc_loss = 0
                if model.seen % params["save_every"] == 0:
                    torch.save(model.state_dict(), os.path.join(params["output_path"], "latest.pt"))
                    pd.DataFrame(log).to_csv(os.path.join(params["output_path"], "latest.csv"))
                if model.seen % params["val_every"] == 0 and val_ds is not None:
                    val(model, val_ds, params, device)
                #     print("Validation: rotamer {:.4f} aa {:.4f}".format(rot_acc, aa_acc))
        output_filename = os.path.join(params["output_path"], "ep_{}.pt".format(str(i + 1).zfill(2)))
        torch.save(model.state_dict(), output_filename)
        print("{} saved. time elapsed: {}s".format(output_filename, time.time() - t1))
        # if val_ds is not None:
        #     rot_acc, aa_acc = val(model, val_ds, params, device)
        #     print("Validation: rotamer {:.4f} aa {:.4f}".format(rot_acc, aa_acc))
        #     with open(os.path.join(params["output_path"], "ep_{}.txt".format(str(i+1).zfill(2))), "w") as f:
        #         f.write("{:.4f},{:.4f}".format(rot_acc, aa_acc))
    return model


def val(model, ds, params, device, batch_stats=False, verbose=False):
    n_iters = int(len(ds) / params["batch_size"])
    dataloader = DataLoader(ds, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"],
                            drop_last=False)
    model = model.to(device)
    if batch_stats:
        model.train()
    else:
        model.eval()
    aaacc = AAAccuracy(matrix_path="rotamer2aa_ukn.npy", device=device)
    log = list()
    seen = 0
    acc_pre = 0
    acc_rec = 0
    acc_rot = 0
    acc_aa = 0
    t1 = time.time()
    with torch.no_grad():
        for j, (x, y) in enumerate(dataloader):
            x = x.to(device)
            o_field = y[0].to(device)
            c_field = y[1].to(device)
            ca_field = y[2].to(device)
            n_field = y[3].to(device)
            rot_labels = y[4].to(device)
            aa_labels = y[5].to(device)
            model.train()
            output = model(x)
            n_gt, n_p, pre, rec, o_err, c_err, ca_err, n_err, rot_cat, aa_cat \
                = metrics(output, o_field.float(), c_field.float(), ca_field.float(), n_field.float(),
                          rot_labels.long(), aa_labels, aaacc=aaacc)
            if verbose:
                print("O {:.3f}  C {:.3f}  CA {:.3f}  N {:.3f}".format(o_err, c_err, ca_err, n_err))
            seen += x.size(0)
            acc_pre += x.size(0) * pre
            acc_rec += x.size(0) * rec
            acc_rot += x.size(0) * rot_cat
            acc_aa += x.size(0) * aa_cat
        acc_pre /= seen
        acc_rec /= seen
        acc_rot /= seen
        acc_aa /= seen
    print("======================================================================")
    print("validation: precision {:.4f}  recall {:.4f}  rotamer {:.4f}  aa {:.4f}"
          .format(acc_pre, acc_rec, acc_rot, acc_aa))
    print("======================================================================")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_df_path", type=str, default=None, help="Path to training data DF")
    p.add_argument("--val_df_path", type=str, default=None, help="Path to validation data DF")
    p.add_argument("--output_path", type=str, default="backup", help="Path to put output files")
    p.add_argument("--model", type=str, default=None, help="Path to a trained model")
    p.add_argument("--seed", type=int, default=2020, help="Random seed for NumPy and Pandas")
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")
    p.add_argument("--n_filters", "-n", type=int, default=16, help="Number of filters in RotamerNet")
    p.add_argument("--n_epoches", "-e", type=int, default=11, help="Number of epoches")
    p.add_argument("--max_size", type=int, default=256, help="Max data dimesion (voxels)")
    p.add_argument("--val_every", type=int, default=32, help="On the fly validation every X steps")
    p.add_argument("--n_val", type=int, default=-1, help="Size of validation dataset")
    p.add_argument("--focalloss", action="store_true", help="Use Focal Loss")
    p.add_argument("--focalloss_gamma", type=float, default=2.0, help="Gamma parameter for FocalLoss")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.00, help="L2 weight decay constant")
    p.add_argument("--cat_weight", action="store_true", help="Use categorical weight")
    p.add_argument("--skip_training", action="store_true", help="Skip training")
    p.add_argument("--transform", action="store_true", help="Enable data augmentation")
    p.add_argument("--normalize", action="store_true", help="Normalize data")
    p.add_argument("--bottleneck", action="store_true", help="Use bottleneck blocks")
    p.add_argument("--batch_stats", action="store_true", help="Use batch stats instead of learned for inference")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    model = AlphaNet(n_filters=args.n_filters, bottleneck=args.bottleneck)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        print("trained model {} loaded".format(args.model))
    if args.verbose:
        print(model)
    params = dict(epochs=args.n_epoches, batch_size=1, num_workers=2, lr=args.lr, print_every=1, back_every=1,
                  save_every=1024, val_every=args.val_every, weight_decay=args.weight_decay,
                  output_path=args.output_path)
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    val_ds = AlphaDataset(args.val_df_path, downsample=8, normalize=args.normalize, max_size=args.max_size,
                          n=args.n_val)
    if args.verbose:
        print("val_ds", len(val_ds))
    if args.skip_training:
        t1 = time.time()
        val(model, val_ds, params, device, verbose=True)
        t2 = time.time()
        print("time elapsed: {:.1f} s".format(t2-t1))
    else:
        train_ds = AlphaDataset(args.train_df_path, downsample=8, normalize=args.normalize, transform=args.transform,
                                max_size=args.max_size)
        if args.verbose:
            print("train_ds", len(train_ds))
        model = train(model, train_ds, params, device=device, val_ds=val_ds)
    # rot_acc, aa_acc = val(model, val_ds, params, device=device, batch_stats=args.batch_stats)
    # print("Validation: rotamer {:.4f} aa {:.4f}".format(rot_acc, aa_acc))


if __name__ == "__main__":
    main()
