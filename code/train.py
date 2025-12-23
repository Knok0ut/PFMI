import numpy as np
import torch
from torch.utils.data import DataLoader

from util.models import get_architecture
from util.code_logger import CodeLogger
from util.trainer import train as trainer
from util.datasets import load_dataset
import argparse
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="alexnet")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--shadow", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
verbose = args.verbose
train_shadow = args.shadow

if train_shadow:
    suffix = "_shadow"
    # suffix = ""
    pre_fix = "attack_"
else:
    suffix = ""
    pre_fix = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cl = CodeLogger(comment=f"train model {pre_fix}{model_name}_{dataset}{suffix}", args=args, file_name=__file__)

model = get_architecture(model_name + '_' + dataset)

train_set, test_set = load_dataset(dataset, [pre_fix + "train", pre_fix + "test"])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

cl.logger.info(f"training for {dataset}_{model_name}{suffix}")
model, optimizer = trainer(model, train_loader, epochs, lr, cl, verbose=verbose)
torch.save({"model": model.state_dict(), "optimizer": optimizer}, f"../checkpoints/{dataset}_{model_name}{suffix}.pt")

cnt = 0
all = 0
true_label = []
pred_label = []
for data, label in train_loader:
    true_label.append(label.detach().cpu().numpy())
    data, label = data.to(device), label.to(device)
    out = torch.argmax(model(data), dim=1)
    pred_label.append(out.detach().cpu().numpy())
    mask = out == label
    all += len(mask)
    cnt += torch.sum(mask)
cl.logger.info(f"train acc: {cnt / all}")
true_label = np.concatenate((true_label))
pred_label = np.concatenate((pred_label))
report = classification_report(true_label, pred_label)
cl.logger.info(report)
