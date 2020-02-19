import os
import argparse
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet.unet_model import UNet
from contactmap_dataset import ContactMapDataset


class ContactNet(object):
    
    def __init__(self, model, params):
        self.model = model
        try:
            print("seen", self.model.seen)
        except AttributeError:
            self.model.seen = 0
        self.params = params
        
    def train(self):
        self.model.train()
        params = self.params
        ds = ContactMapDataset(params["train_h5"], random=True)
        dl = DataLoader(ds, batch_size=params["batch_size"], drop_last=True)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters())
        for i, (X, Y) in enumerate(dl):
            if params["verbose"]:
                print("X", X.size(), "Y", Y.size())
            try:
                scores = self.model(X.float())
            except ValueError:
                print("X", X.size(), "Y", Y.size(), "ValueError")
                continue
            if params["verbose"]:
                print("scores", scores.size())
            loss = loss_fn(scores, Y.float())
            loss.backward()
            self.model.seen += X.size(0)
            if self.model.seen % params["optim_batch_size"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                print("seen {} loss {:3.3f}".format(self.model.seen, loss.data.cpu()))
                if self.model.seen % params["steps_save_model"] == 0:
                    self.save()
                    
                
    def validate(self):
        self.model.eval()
        pass
    
    def predict(self, img, sigmoid=False, numpy=False):
        self.model.eval()
        scores = self.model(img.unsqueeze(0).unsqueeze(0).float()).data
        if sigmoid:
            scores = nn.Sigmoid()(scores)
        if numpy:
            scores = scores.numpy()
        return scores
    
    def save(self):
        dest_file = os.path.join(self.params["backup_path"],
                                 "{}_{}.pt".format(self.params["project_name"], str(self.model.seen).zfill(9)))
        torch.save(self.model, dest_file)
        print("{} saved".format(dest_file))
               
        
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", "-m", type=str, default=None, help="Number of epoches")
    p.add_argument("--params_path", type=str, help="Where to find the params file")
    p.add_argument("--epoches", "-n", type=int, default=1, help="Number of epoches")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()

    
def main():
    args = parse_args()
    params = {"train_h5": "output/contactmap_20200219_train.h5",
              "batch_size": 1, "optim_batch_size": 2,
              "n_input": 1, "n_output": 1, "bilinear": True,
              "backup_path": "backup/", "steps_save_model": 50,
              "project_name": "test",
              "verbose": args.verbose}
    if args.model_path is not None:
        model = torch.load(args.model_path)
    else:
        model = UNet(params["n_input"], params["n_output"], bilinear=params["bilinear"])
    contact_net = ContactNet(model, params)
    for i in range(args.epoches):
        contact_net.train()
    
    
if __name__ == "__main__":
    main()