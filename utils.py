import torch


def accuracy(scores, targets, ignore_index=162):
    val, ind = torch.max(scores, 1)
    acc = (ind[targets != ignore_index].squeeze().long() == targets[targets != ignore_index].squeeze().long()).sum()
    # print("n correct", acc.cpu().item())
    acc = acc.item() / scores.size(0)
    return acc
