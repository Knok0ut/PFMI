import torch
import math
import torch.nn as nn
from util.resnet50 import get_resnet50, ResNet50, Bottleneck


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# 改成resnet50
class NCI_Cifar(ResNet50):
    def __init__(self, Bottleneck=Bottleneck, num_classes=10) -> None:
        super(NCI_Cifar, self).__init__(Bottleneck, num_classes)

    def forward(self, x, return_feature=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        features = out.reshape(x.shape[0], -1)
        out = self.fc(features)
        if return_feature:
            return out, features
        else:
            return out


# class NCI_Cifar(nn.Module):  # 训练 ALexNet
#     def __init__(self, num_classes=10):
#         super(NCI_Cifar, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.fc = nn.Linear(256 * 2 * 2, num_classes)
#
#     def forward(self, x, return_feature=False):
#         x = self.features(x)
#         features = x.view(x.size(0), 256 * 4)
#         x = self.fc(features)
#         if return_feature:
#             return x, features
#         else:
#             return x


# class NCI_MNIST(nn.Module):
#     def __init__(self, num_classes=10):
#         super(NCI_MNIST, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.fc = nn.Linear(256, num_classes)
#
#     def forward(self, x, return_feature=False):
#         x = self.features(x)
#         features = x.view(x.size(0), 256)
#         x = self.fc(features)
#         if return_feature:
#             return x, features
#         else:
#             return x
#
#
# class NCI_STL(nn.Module):
#     def __init__(self, num_classes=10):
#         super(NCI_STL, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.fc = nn.Linear(9216, num_classes)
#
#     def forward(self, x, return_feature=False):
#         x = self.features(x)
#         features = x.view(x.size(0), 9216)
#         x = self.fc(features)
#         if return_feature:
#             return x, features
#         else:
#             return x


class NCI_Epsilon(nn.Module):
    def __init__(self, in_size=2000, classes_num=2):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(NCI_Epsilon, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, classes_num)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        x = self.fc(features)
        if return_feature:
            return x, features
        return x


class NCI_Diabetic(nn.Module):
    def __init__(self, in_size=44, classes_num=2):
        if classes_num is None:
            classes_num = 2
        self.classes_num = classes_num
        super(NCI_Diabetic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, classes_num)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        x = self.fc(features)
        if return_feature:
            return x, features
        return x


class Alexnet_Cifar(nn.Module):  # 训练 ALexNet
    def __init__(self, num_classes=10):
        super(Alexnet_Cifar, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, return_feat=False):
        x = self.features(x)
        features = x.view(x.size(0), 256 * 4)
        x = self.classifier(features)
        if return_feat:
            return x, features
        else:
            return x


class Alexnet_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(Alexnet_MNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_feat=False):
        x = self.features(x)
        features = x.view(x.size(0), 256)
        x = self.classifier(features)
        if return_feat:
            return x, features
        else:
            return x


class Alexnet_STL(nn.Module):
    def __init__(self, num_classes=10):
        super(Alexnet_STL, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, return_feat=False):
        x = self.features(x)
        features = x.view(x.size(0), 9216)
        x = self.classifier(features)
        if return_feat:
            return x, features
        else:
            return x


class Resnet50_Cifar(ResNet50):
    def __init__(self, Bottleneck, num_classes=10) -> None:
        super(Resnet50_Cifar, self).__init__(Bottleneck, num_classes)


class Resnet50_STL(ResNet50):
    def __init__(self, Bottleneck, num_classes=10) -> None:
        super(Resnet50_STL, self).__init__(Bottleneck, num_classes)
        self.fc = nn.Linear(2048, num_classes)


class Resnet50_MNIST(ResNet50):
    def __init__(self, Bottleneck, num_classes=10) -> None:
        super(Resnet50_MNIST, self).__init__(Bottleneck, num_classes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(2048, num_classes)


class NCI_MNIST(Resnet50_MNIST):
    def __init__(self, Bottleneck=Bottleneck, num_classes=10) -> None:
        super(NCI_MNIST, self).__init__(Bottleneck, num_classes)

    def forward(self, x, return_feature=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        features = out.reshape(x.shape[0], -1)
        out = self.fc(features)
        if return_feature:
            return out, features
        else:
            return out


class NCI_STL(Resnet50_STL):
    def __init__(self, Bottleneck=Bottleneck, num_classes=10) -> None:
        super(NCI_STL, self).__init__(Bottleneck, num_classes)

    def forward(self, x, return_feature=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        features = out.reshape(x.shape[0], -1)
        out = self.fc(features)
        if return_feature:
            return out, features
        else:
            return out


class Autoencoder_CNN_32(nn.Module):
    '''
    输入尺寸：[batch, channel, 32, 32]
    使用3x3卷积核
    '''

    def __init__(self, channel=3):
        super(Autoencoder_CNN_32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=3, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # [batch, 48, 4, 4]
        )
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, channel, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 3, 32, 32]
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.relu(encoded)
        decoded = self.decoder(x)
        return encoded, decoded


class Autoencoder_CNN_28(nn.Module):
    '''
    真正适配28x28输入，保证输出也是28x28，不炸成32x32
    '''

    def __init__(self, channel=1):
        super(Autoencoder_CNN_28, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, channel, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            # nn.Sigmoid(),  # 保证输出在0-1之间
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Autoencoder_MLP(nn.Module):
    '''
    https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    '''

    def __init__(self, in_size=2000):
        super(Autoencoder_MLP, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Linear(in_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, in_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.relu(encoded)
        decoded = self.decoder(x)
        return encoded, decoded

class Autoencoder_Diabetic(nn.Module):
    '''
    https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    '''

    def __init__(self, in_size=10):
        super(Autoencoder_Diabetic, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, in_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.relu(encoded)
        decoded = self.decoder(x)
        return encoded, decoded

class MLP(nn.Module):
    def __init__(self, in_size=2000, classes_num=10):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, classes_num)
        )

    def forward(self, x):
        x = self.model(x)
        # if return_feat:
        #     return x, x
        return x


class Diabetic_MLP(nn.Module):
    def __init__(self, in_size=44, classes_num=2):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(Diabetic_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, classes_num)
        )

    def forward(self, x, return_feat=False):
        x = self.model(x)
        if return_feat:
            return x, x
        return x


def get_architecture(model_name):
    # if model_name == "resnet20_cifar10":
    #     return ResNet_Cifar(BasicBlock, [3, 3, 3], 10)
    if model_name == "resnet50_cifar10":
        return get_resnet50()
    elif model_name == "resnet50_fashion":
        return Resnet50_MNIST(Bottleneck)
    elif model_name == "resnet50_stl10":
        return Resnet50_STL(Bottleneck)
    # elif model_name == "resnet20_cifar100":
    #     return ResNet_Cifar(BasicBlock, [3, 3, 3], 100)
    elif model_name == "alexnet_mnist":
        return Alexnet_MNIST()
    elif model_name == "alexnet_fashion":
        return Alexnet_MNIST()
    elif model_name == "alexnet_cifar10":
        return Alexnet_Cifar(10)
    elif model_name == "alexnet_cifar100":
        return Alexnet_Cifar(100)
    elif model_name == "mlp_purchase100":
        return MLP(in_size=600, classes_num=100)
    elif model_name == "mlp_texas100":
        return MLP(in_size=6169, classes_num=100)
    elif model_name == "alexnet_stl10":
        return Alexnet_STL()
    elif model_name == "mlp_epsilon":
        return MLP(in_size=2000, classes_num=2)
    elif model_name == "cifar10_AE":
        return Autoencoder_CNN_32(channel=3)
    elif model_name == "stl10_AE":
        return Autoencoder_CNN_32(channel=3)
    elif model_name == "fashion_AE":
        return Autoencoder_CNN_28(channel=1)
    elif model_name == "epsilon_AE":
        return Autoencoder_MLP(in_size=2000)
    elif model_name == "NCI_cifar10":
        return NCI_Cifar(num_classes=10)
    elif model_name == "NCI_fashion":
        return NCI_MNIST(num_classes=10)
    elif model_name == "NCI_stl10":
        return NCI_STL(num_classes=10)
    elif model_name == "NCI_epsilon":
        return NCI_Epsilon(in_size=2000)
    elif model_name == "mlp_diabetic":
        return MLP(in_size=10)
    elif model_name == "NCI_diabetic":
        return NCI_Epsilon(in_size=10)
    elif model_name == "diabetic_AE":
        return Autoencoder_Diabetic(in_size=10)


class CompletionNet(nn.Module):
    def __init__(self, in_size=300, out_size=300, hidden_size=512):
        self.out_size = out_size
        self.hidded_size = hidden_size
        super(CompletionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def load_model(model, model_path):
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict["model"])
    optimizer = state_dict["optimizer"]
    return model, optimizer
