import os.path

import torch
from torch.utils.data import DataLoader
import argparse

from util.models import get_architecture
from util.datasets import load_dataset
from util.code_logger import CodeLogger
from util.trainer import train_AE as trainer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--max_dis", type=float, default=10)
parser.add_argument("--lamb", type=float, default=0.01)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
verbose = args.verbose
max_dis = args.max_dis
lamb = args.lamb

cl = CodeLogger(comment=f"train ae for {dataset}", args=args, file_name=__file__)

model = get_architecture(f"{dataset}_AE").cuda()
train_set = load_dataset(dataset, ["aux3"])[0]
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model, optimizer = trainer(model, train_loader, epochs=epochs, lr=lr, max_dis=max_dis, lamb=lamb, cl=cl,
                           verbose=verbose)

if not os.path.exists("../checkpoints/AE/"):
    os.mkdir("../checkpoints/AE/")
torch.save({"model": model.state_dict(), "optimizer": optimizer}, f"../checkpoints/AE/{dataset}_AE.pt")
