import argparse

import torch
from torch.utils.data import DataLoader

from util.util import *
from util.code_logger import CodeLogger
from util.datasets import Purchase100, Texas100, Adult, CIFAR_10, load_dataset
from util.models import get_architecture, load_model
import numpy as np
import shap
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="alexnet")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--_file_name", type=str, default=__file__)

args = parser.parse_args()

model_name = args.model
dataset = args.dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cl = CodeLogger(comment=f"evaluate attack {model_name}_{dataset}", args=args, file_name=__file__)

if __name__ == "__main__":
    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    train_set, test_set = load_dataset(dataset, ["member", "non_member"])
    data = train_set.data
    background_img = data[:16]
    background_img = background_img.contiguous()
    train_set.data = data[16:]
    train_set.label = train_set.label[16:]

    test_set.data = test_set.data[16:]
    test_set.label = test_set.label[16:]

    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    train_loader = DataLoader(train_set, shuffle=False, batch_size=1024)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1024)

    # background_img, background_label = next(iter(val_loader))
    #
    explainer = shap.DeepExplainer(model, background_img.cuda())

    datas = []
    labels = []
    shaps = []

    for data, label in train_loader:
        data = data.cuda()
        sp = explainer.shap_values(data, check_additivity=False)
        datas.append(data.detach().cpu())
        labels.append(label.cpu())
        shaps.append(torch.from_numpy(sp))

    datas = torch.cat(datas, dim=0)
    labels = torch.cat(labels)
    shaps = torch.cat(shaps, dim=0)

    torch.save({"datas": datas, "labels": labels, "shaps": shaps}, f"../data/{dataset}/shap_train.pt")

    datas = []
    labels = []
    shaps = []

    for data, label in test_loader:
        data = data.cuda()
        sp = explainer.shap_values(data, check_additivity=False)
        datas.append(data.detach().cpu())
        labels.append(label.cpu())
        shaps.append(torch.from_numpy(sp))

    datas = torch.cat(datas, dim=0)
    labels = torch.cat(labels)
    shaps = torch.cat(shaps, dim=0)

    torch.save({"datas": datas, "labels": labels, "shaps": shaps}, f"../data/{dataset}/shap_test.pt")
