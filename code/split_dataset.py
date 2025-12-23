'''
split dataset
'''

import os
from argparse import ArgumentParser

import numpy as np
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

aux_num = 5

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")

args = parser.parse_args()
dataset = args.dataset

# 加载数据
if dataset == "cifar10":
    cifar_set = torchvision.datasets.CIFAR10(root="../data/cifar10/", download=True, transform=T.ToTensor())
    data = cifar_set.data
    labels = np.array(cifar_set.targets)
    packet_len = len(data) // 10

elif dataset == "stl10":
    stl = torchvision.datasets.STL10(root="../data/stl10/", split="train", download=True)
    stl_test = torchvision.datasets.STL10(root="../data/stl10/", split="test", download=True)
    data = np.concatenate([stl.data, stl_test.data], axis=0).astype(float)
    labels = np.concatenate([stl.labels, stl_test.labels]).astype(int)
    packet_len = len(data) // 10

elif dataset == "epsilon":
    data = np.loadtxt("../data/epsilon/train.tsv", float, delimiter="\t")
    labels = data[:, 0].astype(int)
    data = data[:, 1:]
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    data = (data - mean) / (std + 1e-6)
    labels[labels == -1] = 0
    # packet_len = len(data) // 40
    packet_len = 5000

elif dataset == "fashion":
    fashion_set = torchvision.datasets.FashionMNIST(root="../data/fashion/", download=True)
    data = fashion_set.data
    labels = np.array(fashion_set.targets)
    packet_len = len(data) // 10

data_remaining, train_data, labels_remaining, train_label = train_test_split(
    data, labels, test_size=packet_len, stratify=labels
)
data_remaining, test_data, labels_remaining, test_label = train_test_split(
    data_remaining, labels_remaining, test_size=packet_len, stratify=labels_remaining
)
data_remaining, shadow_train_data, labels_remaining, shadow_train_label = train_test_split(
    data_remaining, labels_remaining, test_size=packet_len, stratify=labels_remaining
)
data_remaining, shadow_test_data, labels_remaining, shadow_test_label = train_test_split(
    data_remaining, labels_remaining, test_size=packet_len, stratify=labels_remaining
)

if not os.path.exists(f"../data/{dataset}"):
    os.mkdir(f"../data/{dataset}")

# 存储训练数据和标签
np.save(f"../data/{dataset}/train_data.npy", train_data)
np.save(f"../data/{dataset}/train_label.npy", train_label)

# 存储测试数据和标签
np.save(f"../data/{dataset}/test_data.npy", test_data)
np.save(f"../data/{dataset}/test_label.npy", test_label)

# 存储攻击训练数据和标签
np.save(f"../data/{dataset}/attack_train_data.npy", shadow_train_data)
np.save(f"../data/{dataset}/attack_train_label.npy", shadow_train_label)

# 存储攻击测试数据和标签
np.save(f"../data/{dataset}/attack_test_data.npy", shadow_test_data)
np.save(f"../data/{dataset}/attack_test_label.npy", shadow_test_label)

# 存储成员数据和标签
np.save(f"../data/{dataset}/member_data.npy", train_data)
np.save(f"../data/{dataset}/member_label.npy", train_label)

# 存储非成员数据和标签
np.save(f"../data/{dataset}/non_member_data.npy", test_data)
np.save(f"../data/{dataset}/non_member_label.npy", test_label)

# 存储额外数据和标签
for i in range(aux_num):
    if len(data_remaining) > packet_len:
        data_remaining, aux_data, labels_remaining, aux_label = train_test_split(
            data_remaining, labels_remaining, test_size=packet_len, stratify=labels_remaining
        )
        np.save(f"../data/{dataset}/aux{i}_data.npy", aux_data)
        np.save(f"../data/{dataset}/aux{i}_label.npy", aux_label)
    else:
        np.save(f"../data/{dataset}/aux{i}_data.npy", data_remaining)
        np.save(f"../data/{dataset}/aux{i}_label.npy", labels_remaining)
        print(f"early stop at aux{i}")
        break
