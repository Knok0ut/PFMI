import argparse
from torch.utils.data import DataLoader

from util.util import *
from util.code_logger import CodeLogger
from util.datasets import Purchase100, Texas100, Adult, CIFAR_10, load_dataset
from util.models import get_architecture, load_model
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def mad_score(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    return -x + 1


def load_data(data_path):
    data = np.load(data_path)
    label = data[:, -1]
    data = data[:, :-1]
    return data, label


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--generate_epochs", type=int, default=1)
parser.add_argument("--generate_lr", type=float, default=100)
parser.add_argument("--plot_flag", action="store_true")
parser.add_argument("--thresh", type=float, default=64)
parser.add_argument("--known_feature_num", type=int, default=800)
parser.add_argument("--grad_norm", action="store_true")
parser.add_argument("--case_study", action="store_true")
parser.add_argument("--_file_name", type=str, default=__file__)

args = parser.parse_args()

model_name = args.model
dataset = args.dataset
generate_epochs = args.generate_epochs
generate_lr = args.generate_lr
plot_flag = args.plot_flag
thresh = args.thresh
known_features_num = args.known_feature_num
grad_norm = args.grad_norm
case_study = args.case_study

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cl = CodeLogger(comment=f"evaluate attack {model_name}_{dataset}", args=args, file_name=__file__)

if __name__ == "__main__":
    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    shadow_model = get_architecture(model_name + '_' + dataset).to(device)
    shadow_model, _ = load_model(shadow_model, f"../checkpoints/{dataset}_{model_name}_shadow.pt")

    train_set, test_set, val_set, other_set = load_dataset(dataset,
                                                           ["member", "non_member", "aux1", "aux2"])
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    print(f"-------------------train_set len: {len(train_set)}, test_set len: {len(test_set)}-------------------------")

    if case_study:
        mask = torch.tensor(~np.load("../data/diabetic/mask.npy"))
        known_features_num = 4
    else:
        mask = generate_mask(mask_shape, unknown_num=f_num - known_features_num)
    # train_set, test_set = init_dataset(train_set, test_set, mask=mask)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    # 生成攻击测试样本
    cl.logger.info(f"known_features_num: {known_features_num}, generate_epochs: {generate_epochs}")

    gen_train_data, gen_train_label = generate_data_maskd(train_loader, model, generate_lr=generate_lr,
                                                          generate_epochs=generate_epochs, mask=mask,
                                                          grad_norm=grad_norm)
    gen_test_data, gen_test_label = generate_data_maskd(test_loader, model, generate_lr=generate_lr,
                                                        generate_epochs=generate_epochs, mask=mask, grad_norm=grad_norm)

    gen_train_data = get_pred_vec(shadow_model, gen_train_data.cuda()).cpu()
    gen_test_data = get_pred_vec(shadow_model, gen_test_data.cuda()).cpu()
    other_train_data = get_pred_vec(shadow_model, other_set.data.cuda()).cpu()
    # gen_train_data = gen_train_data.numpy()
    # gen_test_data = gen_test_data.numpy()
    #
    # gen_train_label = gen_train_label.numpy()
    # gen_test_label = gen_test_label.numpy()
    #
    # other_train_data = other_set.data.numpy()
    other_train_label = other_set.label.numpy()

    # reshape
    gen_train_data = gen_train_data.reshape((len(gen_train_data), -1))[gen_train_label == 0]
    gen_test_data = gen_test_data.reshape((len(gen_test_data), -1))[gen_test_label == 0]
    other_train_data = other_train_data.reshape((len(other_train_data), -1))[other_train_label == 0]

    np.random.shuffle(gen_train_data)
    np.random.shuffle(gen_test_data)
    np.random.shuffle(other_train_data)

    max_num = 50
    gen_train_data = gen_train_data[:max_num]
    gen_test_data = gen_test_data[:max_num]
    other_train_data = other_train_data[:max_num]

    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(np.concatenate([gen_train_data, gen_test_data, other_train_data]))
    colors = ["red", "blue", "green"]
    names = ["train", "test", "other"]
    for i, (color, name) in enumerate(zip(colors, names)):
        plot_x = x_tsne[i * max_num:i * max_num + max_num]
        plt.scatter(plot_x[:, 0], plot_x[:, 1], c=color, label=name)
    plt.legend()
    plt.show()

    # train_mad_ls, test_mad_ls = cal_mad(gen_train_data, gen_train_label, gen_test_data, gen_test_label,
    #                                     other_train_data, other_train_label, classes)
    # gen_shadow_train_data, gen_shadow_train_label = generate_data_maskd(train_loader, shadow_model,
    #                                                                     generate_lr=generate_lr,
    #                                                                     generate_epochs=generate_epochs, mask=mask,
    #                                                                     grad_norm=grad_norm)
    # gen_shadow_test_data, gen_shadow_test_label = generate_data_maskd(test_loader, shadow_model,
    #                                                                   generate_lr=generate_lr,
    #                                                                   generate_epochs=generate_epochs, mask=mask,
    #                                                                   grad_norm=grad_norm)
    #
    # gen_shadow_train_data = gen_shadow_train_data.numpy()
    # gen_shadow_test_data = gen_shadow_test_data.numpy()
    #
    # gen_shadow_train_label = gen_shadow_train_label.numpy()
    # gen_shadow_test_label = gen_shadow_test_label.numpy()
    #
    # # reshape
    # gen_shadow_train_data = gen_shadow_train_data.reshape((len(gen_shadow_train_data), -1))
    # gen_shadow_test_data = gen_shadow_test_data.reshape((len(gen_shadow_test_data), -1))
    #
    # shadow_train_mad_ls, shadow_test_mad_ls = cal_mad(gen_shadow_train_data, gen_shadow_train_label,
    #                                                   gen_shadow_test_data, gen_shadow_test_label, other_train_data,
    #                                                   other_train_label, classes)
    #
    # assert len(train_mad_ls) == len(test_mad_ls), "train,test length not equal"
    # assert len(shadow_train_mad_ls) == len(shadow_test_mad_ls), "shadow_train,shadow_test length not equal"
    #
    # # 去掉nan
    # train_nan_mask = np.bitwise_or(np.isnan(train_mad_ls), np.isnan(shadow_train_mad_ls))
    # test_nan_mask = np.bitwise_or(np.isnan(test_mad_ls), np.isnan(shadow_test_mad_ls))
    # train_mad_ls, shadow_train_mad_ls = train_mad_ls[~train_nan_mask], shadow_train_mad_ls[~train_nan_mask]
    # test_mad_ls, shadow_test_mad_ls = test_mad_ls[~test_nan_mask], shadow_test_mad_ls[~test_nan_mask]
    #
    # train_mad_ls, shadow_train_mad_ls = train_mad_ls[:len(test_mad_ls)], shadow_train_mad_ls[:len(test_mad_ls)]
    # test_mad_ls, shadow_test_mad_ls = test_mad_ls[:len(train_mad_ls)], shadow_test_mad_ls[:len(train_mad_ls)]
    #
    # effective_len = len(train_mad_ls)
    # print(f"有效的样本数量：{effective_len}")
    #
    # # 计算分数
    # train_score = shadow_train_mad_ls / (train_mad_ls + 1e-8)
    # test_score = shadow_test_mad_ls / (test_mad_ls + 1e-8)
    #
    # fpr, tpr, threshold = roc_curve(np.array([1] * len(train_score) + [0] * len(test_score)),
    #                                 np.concatenate([train_score, test_score]))
    # auc_score = auc(fpr, tpr)
    # print(f"auc: {float(auc_score)}")
