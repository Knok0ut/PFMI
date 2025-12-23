import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image


class Purchase100(Dataset):
    def __init__(self, kind="train"):
        super(Purchase100, self).__init__()
        self.kind = kind
        self.data, self.label = self.load_data()

    def load_data(self):
        data = np.load(f"../data/texas100/{self.kind}_data.npy")
        label = np.load(f"../data/texas100/{self.kind}_label.npy").astype(int)
        label = torch.tensor(label).long()
        data = torch.tensor(data).float()
        return data, label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class Texas100(Dataset):
    def __init__(self, kind="train"):
        super(Texas100, self).__init__()
        self.kind = kind
        self.data, self.label = self.load_data()

    def load_data(self):
        data = np.load(f"../data/texas100/{self.kind}_data.npy")
        label = np.load(f"../data/texas100/{self.kind}_label.npy").astype(int)
        label = torch.tensor(label).long()
        data = torch.tensor(data).float()
        return data, label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class Adult(Dataset):
    def __init__(self, kind="train"):
        super(Adult, self).__init__()
        self.kind = kind
        self.data, self.label = self.load_data()

    def load_data(self):
        data = np.load(f"../data/adult/{self.kind}.npy")
        label = torch.tensor(data[:, -1].astype(int)).long()
        data = torch.tensor(data[:, :-1]).float()
        return data, label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, kind="train"):
        self.mean = 0.1307
        self.std = 0.3081
        self.data, self.label = self.load_data()
        self.standardize()

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/mnist/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/mnist/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        label = torch.tensor(label).long()
        return data, label

    def standardize(self):
        """使用预定义的 mean 和 std 进行标准化"""
        self.data = (self.data / 255.0 - self.mean) / self.std  # 归一化并标准化

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class FASHION(torchvision.datasets.MNIST):
    def __init__(self, kind="train", c=-1):
        self.kind = kind
        self.mean = 0.2860
        self.std = 0.3530
        self.data, self.label = self.load_data()
        if c != -1:
            mask = self.label == c
            self.data = self.data[mask]
            self.label = self.label[mask]
        self.standardize()

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/fashion/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/fashion/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        label = torch.tensor(label).long()

        data = data.reshape((len(data), 1, 28, 28))
        return data, label

    def standardize(self):
        """使用预定义的 mean 和 std 进行标准化"""
        self.data = (self.data / 255.0 - self.mean) / self.std  # 归一化并标准化

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class CIFAR_10(Dataset):
    def __init__(self, kind="train", mean=None, std=None, c=-1):
        super(CIFAR_10, self).__init__()
        self.kind = kind

        # 使用默认的 CIFAR-10 统计值
        self.mean = torch.tensor(mean if mean else [0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor(std if std else [0.2470, 0.2435, 0.2616]).view(3, 1, 1)

        self.data, self.label = self.load_data()
        self.standardize()
        if c != -1:
            mask = self.label == c
            self.data = self.data[mask]
            self.label = self.label[mask]

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/cifar10/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/cifar10/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        assert data.shape[3] == 3  # 确保通道数为 3
        data = data.permute(0, 3, 1, 2)  # 变换为 (N, C, H, W)

        label = torch.tensor(label).long()
        return data, label

    def standardize(self):
        """使用预定义的 mean 和 std 进行标准化"""
        self.data = (self.data / 255.0 - self.mean) / self.std  # 归一化并标准化

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class FCIFAR_10(Dataset):
    def __init__(self, kind="train", mean=None, std=None, c=-1):
        super(FCIFAR_10, self).__init__()
        self.kind = kind

        # 使用默认的 CIFAR-10 统计值
        self.mean = torch.tensor(mean if mean else [0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor(std if std else [0.2470, 0.2435, 0.2616]).view(3, 1, 1)

        self.data, self.label = self.load_data()
        self.standardize()
        self.data = self.data.reshape((len(self.data), -1))
        if c != -1:
            mask = self.label == c
            self.data = self.data[mask]
            self.label = self.label[mask]

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/cifar10/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/cifar10/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        assert data.shape[3] == 3  # 确保通道数为 3
        data = data.permute(0, 3, 1, 2)  # 变换为 (N, C, H, W)

        label = torch.tensor(label).long()
        return data, label

    def standardize(self):
        """使用预定义的 mean 和 std 进行标准化"""
        self.data = (self.data / 255.0 - self.mean) / self.std  # 归一化并标准化

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class ExpendDataset(Dataset):
    def __init__(self, original_dataset, k, loc=0, std=0.1):
        """
        Args:
            original_dataset (Dataset): The original dataset to augment.
            k (int): The number of times each sample is duplicated.
        """
        import torch.distributions as distributions
        self.noise_sampler = distributions.Normal(loc, std)
        self.original_dataset = original_dataset
        self.k = k

    def __len__(self):
        return len(self.original_dataset) * self.k

    def __getitem__(self, idx):
        # Map the global index to the original dataset's index
        original_idx = idx // self.k
        data, label = self.original_dataset[original_idx]
        return data + self.noise_sampler.sample(data.shape), label


class CIFAR_100(Dataset):
    def __init__(self, kind="train", mean=None, std=None):
        super(CIFAR_100, self).__init__()
        self.kind = kind

        # 使用默认的 CIFAR-10 统计值
        self.mean = torch.tensor(mean if mean else [0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        self.std = torch.tensor(std if std else [0.2675, 0.2565, 0.2761]).view(3, 1, 1)

        self.data, self.label = self.load_data()
        self.standardize()

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/cifar100/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/cifar100/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        assert data.shape[3] == 3  # 确保通道数为 3
        data = data.permute(0, 3, 1, 2)  # 变换为 (N, C, H, W)

        label = torch.tensor(label).long()
        return data, label

    def standardize(self):
        """使用预定义的 mean 和 std 进行标准化"""
        self.data = (self.data / 255.0 - self.mean) / self.std  # 归一化并标准化

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class STL_10(Dataset):
    def __init__(self, kind="train", mean=None, std=None, c=-1):
        super(STL_10, self).__init__()
        self.kind = kind
        self.mean = torch.tensor(mean if mean else [0.4408, 0.4279, 0.3867]).view(3, 1, 1)
        self.std = torch.tensor(std if std else [0.2682, 0.2610, 0.2686]).view(3, 1, 1)

        self.data, self.label = self.load_data()
        if c != -1:
            mask = self.label == c
            self.data = self.data[mask]
            self.label = self.label[mask]
        self.standardize()

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/stl10/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/stl10/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        assert data.shape[1] == 3  # 确保通道数为 3
        # data = data.permute(0, 3, 1, 2)  # 变换为 (N, C, H, W)

        label = torch.tensor(label).long()
        return data, label

    def standardize(self):
        """使用预定义的 mean 和 std 进行标准化"""
        self.data = (self.data / 255.0 - self.mean) / self.std  # 归一化并标准化

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class Epsilon(Dataset):
    def __init__(self, kind="train", c=-1):
        super(Epsilon, self).__init__()
        self.kind = kind
        self.data, self.label = self.load_data()
        if c != -1:
            mask = self.label == c
            self.data = self.data[mask]
            self.label = self.label[mask]

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/epsilon/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/epsilon/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        label = torch.tensor(label).long()
        return data, label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class Diabetic(Dataset):
    def __init__(self, kind="train", c=-1):
        super(Diabetic, self).__init__()
        self.kind = kind
        self.data, self.label = self.load_data()
        if c != -1:
            mask = self.label == c
            self.data = self.data[mask]
            self.label = self.label[mask]

    def load_data(self):
        """加载数据并转换为张量"""
        data = np.load(f"../data/diabetic/{self.kind}_data.npy")  # 形状 (N, H, W, C)
        label = np.load(f"../data/diabetic/{self.kind}_label.npy")

        data = torch.tensor(data).float()
        label = torch.tensor(label).long()
        return data, label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


def load_dataset(dataset_name, file_names: list, c=-1):
    DS = None
    ret = []
    if dataset_name == "cifar10":
        DS = CIFAR_10
    elif dataset_name == "purchase100":
        DS = Purchase100
    elif dataset_name == "texas100":
        DS = Texas100
    elif dataset_name == "adult":
        DS = Adult
    elif dataset_name == "fcifar10":
        DS = FCIFAR_10
    elif dataset_name == "cifar100":
        DS = CIFAR_100
    elif dataset_name == "fashion":
        DS = FASHION
    elif dataset_name == "stl10":
        DS = STL_10
    elif dataset_name == "epsilon":
        DS = Epsilon
    elif dataset_name == "diabetic":
        DS = Diabetic
    for file_name in file_names:
        ret.append(DS(file_name, c=c))
    return ret
