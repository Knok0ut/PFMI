import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
from util.datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur

# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# transform_resnet18 = transforms.Compose([
#     transforms.Lambda(lambda x: TF.resize(x, 224, interpolation=transforms.InterpolationMode.BICUBIC)),
#     transforms.Lambda(lambda x: TF.center_crop(x, 224))
# ])

# moco_transform = transforms.Compose([
#     # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.RandomApply([GaussianBlur(kernel_size=11, sigma=(0.1, 2.0))], p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset  # 原始数据集（如 CIFAR_10）
        self.transform = transform  # 你想要套在上面的 transform
        # self.data = self.base.data
        # self.label = self.base.label

    def __getitem__(self, idx):
        x, y = self.base[idx]  # 拿出原始数据
        return self.transform(x), y  # 应用 transform 到图像，返回 label 不变

    def __len__(self):
        return len(self.base)


# class Transform:
#     def __init__(self):
#         self.moco_transform = transforms.Compose([
#             # transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur(kernel_size=11, sigma=(0.1, 2.0))], p=0.5),
#             transforms.RandomHorizontalFlip(),
#         ])
#
#     def __call__(self, x):
#         x_1 = self.moco_transform(x)
#         x_2 = self.moco_transform(x)
#         return x_1, x_2

class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)  # 更弱的颜色抖动
            ], p=0.8),  # 降低应用概率
            transforms.RandomGrayscale(p=0.1),  # 更少灰度转换
            transforms.RandomApply([
                GaussianBlur(kernel_size=3, sigma=(0.05, 0.5))  # 更轻的模糊
            ], p=0.5),  # 降低概率
            transforms.RandomHorizontalFlip(p=0.3)  # 降低翻转概率
        ])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class TabularTransform:
    def __init__(self, noise_std=0.02, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std       # 高斯噪声标准差
        self.scale_range = scale_range   # 随机缩放范围

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def random_scale(self, x):
        scale = torch.empty_like(x).uniform_(*self.scale_range)
        return x * scale

    def transform_once(self, x):
        x = self.add_noise(x)
        x = self.random_scale(x)
        return x

    def __call__(self, x):
        x1 = self.transform_once(x)
        x2 = self.transform_once(x)
        return x1, x2


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 152:
            self.backbone = models.resnet152()
        else:
            self.backbone = models.resnet18()
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


def freeze_parameters(model, backbone, train_fc=False):
    if backbone == "alexnet":
        for p in model.classifier.parameters():
            p.requires_grad = False
    return model
    # if not train_fc:
    #     for p in model.classifier.parameters():
    #         p.requires_grad = False
    # if backbone == 152:
    #     for p in model.conv1.parameters():
    #         p.requires_grad = False
    #     for p in model.bn1.parameters():
    #         p.requires_grad = False
    #     for p in model.layer1.parameters():
    #         p.requires_grad = False
    #     for p in model.layer2.parameters():
    #         p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance using sklearn as a replacement for FAISS
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbours, metric='euclidean')
    neigh.fit(train_set)
    distances, _ = neigh.kneighbors(test_set, return_distance=True)
    return np.sum(distances ** 2, axis=1)  # FAISS uses squared L2 distance


def get_loaders(dataset, label_class, batch_size, backbone):
    trainset, trainset_1 = load_dataset(dataset, ["aux1", "aux1"])
    testset = load_dataset(dataset, ["aux2"])[0]
    testset.label = torch.tensor([int(t != label_class) for t in testset.label]).long()

    if dataset in ["cifar10", "stl10", "fashion"]:
        trainset_1 = TransformedDataset(trainset_1, Transform())

    elif dataset == "epsilon" or "diabetic":
        trainset_1 = TransformedDataset(trainset_1, TabularTransform())
    else:
        print('Unsupported Dataset')
        exit()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              drop_last=False)
    return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
                                                                  shuffle=True, num_workers=2, drop_last=False)
