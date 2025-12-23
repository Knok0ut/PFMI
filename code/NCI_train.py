import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader

from util.models import get_architecture
from util.code_logger import CodeLogger
from util.trainer import train as trainer
from util.datasets import CIFAR_10, MNIST, Purchase100, Texas100, Adult, FCIFAR_10, load_dataset
import argparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
verbose = args.verbose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cl = CodeLogger(comment=f"train NCI model for {dataset}_", args=args, file_name=__file__)

model = get_architecture("NCI" + '_' + dataset)

train_set, test_set = load_dataset(dataset, ["attack_train", "attack_test"])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

cl.logger.info(f"training for {dataset}_NCI")
model, optimizer = trainer(model, train_loader, epochs, lr, cl, verbose=verbose)
torch.save({"model": model.state_dict(), "optimizer": optimizer}, f"../checkpoints/{dataset}_NCI.pt")
