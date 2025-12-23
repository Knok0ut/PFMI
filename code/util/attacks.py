import os

import torch.cuda
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import numpy as np

from util.util import *
from util.datasets import load_dataset
from util.models import get_architecture, load_model
from util.MSAD import knn_score
from util.NCI import *
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_CADE_encoding(ae, data_loader, numpy_flag=False):
    labels = []
    enc_outs = []
    for data, label in data_loader:
        data = data.cuda()
        enc_out, dec_out = ae(data)
        enc_outs.append(enc_out.detach().cpu().reshape((len(data), -1)))
        labels.append(label)
    if numpy_flag:
        return torch.row_stack(enc_outs).numpy(), torch.cat(labels).numpy()
    return torch.row_stack(enc_outs), torch.cat(labels)


def get_MSAD_encoding(model_name, dataset, data, label, numpy_flag=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    label = label.to(device)

    enc_outs = []
    labels_out = []

    unique_labels = torch.unique(label)

    for c in unique_labels:
        # 找到当前类别 c 对应的所有样本
        mask = (label == c)
        data_c = data[mask].clone().contiguous()  # 取出所有属于c类的数据
        label_c = label[mask]

        if len(data_c) == 0:
            continue

        # 加载当前类别 c 对应的模型
        model = get_architecture(model_name + '_' + dataset).to(device)
        model.load_state_dict(torch.load(f"../checkpoints/MSAD/{dataset}_{model_name}_{c.item()}.pt"))
        model.eval()

        # 批量推理
        with torch.no_grad():
            enc_out = model(data_c)
            enc_outs.append(enc_out.cpu().reshape(len(data_c), -1))
            labels_out.append(label_c.cpu())

    enc_outs = torch.cat(enc_outs, dim=0)
    labels_out = torch.cat(labels_out, dim=0)

    if numpy_flag:
        return enc_outs.numpy(), labels_out.numpy()
    return enc_outs, labels_out


def CADE_attack(dataset, model_name, generate_epochs=1, generate_lr=100, known_percentage="loo", black_box=False,
                init="zero", mask=None):
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    AE = get_architecture(f"{dataset}_AE").to(device)
    AE, _ = load_model(AE, f"../checkpoints/AE/{dataset}_AE.pt")

    train_set, test_set, other_set = load_dataset(dataset, ["member", "non_member", "aux0"])
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 创建结果存储文件夹
    outcome_dir = f"../outcome/performance_eval/CADE/{dataset}_{model_name}"
    os.makedirs(outcome_dir, exist_ok=True)

    # for known_percentage in range(10, 101, 10):
    # for known_percentage in list(range(10, 99, 10)) + ['loo']:  # 10,20,30,...,90,leave-one-out
    if known_percentage == 'loo':
        known_feature_num = f_num - 1
        unknown_feature_num = 1
    else:
        known_feature_num = int(f_num * (known_percentage / 100))
        unknown_feature_num = f_num - known_feature_num

    print(f"known_feature_num: {known_feature_num}/{f_num}")

    if mask is None:
        mask = generate_mask(mask_shape, unknown_num=unknown_feature_num)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    # 生成攻击样本
    if black_box:
        gen_train_data, gen_train_label = generate_data_black_box(train_loader, model, generate_lr=generate_lr,
                                                                  generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_black_box(test_loader, model, generate_lr=generate_lr,
                                                                generate_epochs=generate_epochs, mask=mask, init=init)
    else:
        gen_train_data, gen_train_label = generate_data_maskd(train_loader, model, generate_lr=generate_lr,
                                                              generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_maskd(test_loader, model, generate_lr=generate_lr,
                                                            generate_epochs=generate_epochs, mask=mask, init=init)

    mem_set = TensorDataset(gen_train_data, gen_train_label)
    non_set = TensorDataset(gen_test_data, gen_test_label)
    mem_loader = DataLoader(mem_set, shuffle=False, batch_size=256)
    non_loader = DataLoader(non_set, shuffle=False, batch_size=256)
    other_loader = DataLoader(other_set, shuffle=False, batch_size=256)
    mem_enc, mem_label = get_CADE_encoding(AE, mem_loader, numpy_flag=True)
    non_enc, non_label = get_CADE_encoding(AE, non_loader, numpy_flag=True)
    other_enc, other_label = get_CADE_encoding(AE, other_loader, numpy_flag=True)
    mem_enc = mem_enc.reshape((len(mem_enc), -1))
    non_enc = non_enc.reshape((len(non_enc), -1))
    other_enc = other_enc.reshape((len(other_enc), -1))

    train_mad_ls, test_mad_ls = cal_mad(mem_enc, mem_label,
                                        non_enc, non_label,
                                        other_enc, other_label,
                                        classes)

    fpr, tpr, thresholds = roc_curve(
        np.array([1] * len(train_mad_ls) + [0] * len(test_mad_ls)),
        np.concatenate([-train_mad_ls, -test_mad_ls])
    )

    return fpr, tpr, thresholds


def MSAD_attack(dataset, model_name, generate_epochs=1, generate_lr=100, known_percentage="loo", black_box=False,
                init="zero", mask=None):
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    train_set, test_set, other_set = load_dataset(dataset, ["member", "non_member", "aux0"])
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 创建结果存储文件夹
    outcome_dir = f"../outcome/performance_eval/MSAD/{dataset}_{model_name}"
    os.makedirs(outcome_dir, exist_ok=True)

    # for known_percentage in list(range(10, 99, 10)) + ['loo']:  # 10,20,30,...,90,leave-one-out
    if known_percentage == 'loo':
        known_feature_num = f_num - 1
        unknown_feature_num = 1
    else:
        known_feature_num = int(f_num * (known_percentage / 100))
        unknown_feature_num = f_num - known_feature_num

    print(f"known_feature_num: {known_feature_num}/{f_num}")

    if mask is None:
        mask = generate_mask(mask_shape, unknown_num=unknown_feature_num)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    # 生成攻击样本
    if black_box:
        gen_train_data, gen_train_label = generate_data_black_box(train_loader, model, generate_lr=generate_lr,
                                                                  generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_black_box(test_loader, model, generate_lr=generate_lr,
                                                                generate_epochs=generate_epochs, mask=mask, init=init)
    else:
        gen_train_data, gen_train_label = generate_data_maskd(train_loader, model, generate_lr=generate_lr,
                                                              generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_maskd(test_loader, model, generate_lr=generate_lr,
                                                            generate_epochs=generate_epochs, mask=mask, init=init)

    mem_enc, mem_label = get_MSAD_encoding(model_name, dataset, gen_train_data, gen_train_label, numpy_flag=True)
    non_enc, non_label = get_MSAD_encoding(model_name, dataset, gen_test_data, gen_test_label, numpy_flag=True)
    other_enc, other_label = get_MSAD_encoding(model_name, dataset, other_set.data, other_set.label, numpy_flag=True)

    train_mad_ls = knn_score(other_enc, mem_enc)
    test_mad_ls = knn_score(other_enc, non_enc)

    fpr, tpr, thresholds = roc_curve(
        np.array([1] * len(train_mad_ls) + [0] * len(test_mad_ls)),
        np.concatenate([-train_mad_ls, -test_mad_ls])
    )
    return fpr, tpr, thresholds


def NCI_attack(dataset, model_name, generate_epochs=1, generate_lr=100, known_percentage="loo",
               black_box=False, init="zero", mask=None):
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    train_set, test_set, other_set = load_dataset(dataset, ["member", "non_member", "aux0"])
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)
    other_loader = DataLoader(other_set, batch_size=256, shuffle=False)
    shadow_model = get_architecture('NCI' + '_' + dataset).to(device)
    shadow_model, _ = load_model(shadow_model, f"../checkpoints/{dataset}_NCI.pt")
    processor = NCIPostprocessor(None)
    processor.setup(shadow_model, other_loader, None)

    # 创建结果存储文件夹
    outcome_dir = f"../outcome/performance_eval/NCI/{dataset}_{model_name}"
    os.makedirs(outcome_dir, exist_ok=True)

    # for known_percentage in range(10, 101, 10):
    if known_percentage == 'loo':
        known_feature_num = f_num - 1
        unknown_feature_num = 1
    else:
        known_feature_num = int(f_num * (known_percentage / 100))
        unknown_feature_num = f_num - known_feature_num

    print(f"known_feature_num: {known_feature_num}/{f_num}")
    if mask is None:
        mask = generate_mask(mask_shape, unknown_num=unknown_feature_num)

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    # 生成攻击样本
    if black_box:
        gen_train_data, gen_train_label = generate_data_black_box(train_loader, model, generate_lr=generate_lr,
                                                                  generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_black_box(test_loader, model, generate_lr=generate_lr,
                                                                generate_epochs=generate_epochs, mask=mask, init=init)
    else:
        gen_train_data, gen_train_label = generate_data_maskd(train_loader, model, generate_lr=generate_lr,
                                                              generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_maskd(test_loader, model, generate_lr=generate_lr,
                                                            generate_epochs=generate_epochs, mask=mask, init=init)
    _, train_dis = processor.postprocess(shadow_model, gen_train_data.cuda())
    _, test_dis = processor.postprocess(shadow_model, gen_test_data.cuda())

    fpr, tpr, thresholds = roc_curve(
        np.array([1] * len(train_dis) + [0] * len(test_dis)),
        np.concatenate([-train_dis.detach().cpu().numpy(), -test_dis.detach().cpu().numpy()])
    )

    return fpr, tpr, thresholds


def MRAD(dataset, model_name, generate_epochs=1, generate_lr=100, known_percentage="loo", grad_norm=False,
         black_box=False, init="zero", mask=None):
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    shadow_model = get_architecture(model_name + '_' + dataset).to(device)
    shadow_model, _ = load_model(shadow_model, f"../checkpoints/{dataset}_{model_name}_shadow.pt")

    train_set, test_set, other_set = load_dataset(dataset, ["member", "non_member", "aux0"])
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # for known_percentage in list(range(10, 99, 10)) + ['loo']:  # 10,20,30,...,90,leave-one-out
    if known_percentage == 'loo':
        known_feature_num = f_num - 1
        unknown_feature_num = 1
    else:
        known_feature_num = int(f_num * (known_percentage / 100))
        unknown_feature_num = f_num - known_feature_num

    print(f"known_feature_num: {known_feature_num}/{f_num}")

    if mask is None:
        mask = generate_mask(mask_shape, unknown_num=unknown_feature_num)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    # 生成攻击样本
    if black_box:
        gen_train_data, gen_train_label = generate_data_black_box(train_loader, model, generate_lr=generate_lr,
                                                                  generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_black_box(test_loader, model, generate_lr=generate_lr,
                                                                generate_epochs=generate_epochs, mask=mask, init=init)
    else:
        gen_train_data, gen_train_label = generate_data_maskd(train_loader, model, generate_lr=generate_lr,
                                                              generate_epochs=generate_epochs, mask=mask, init=init)
        gen_test_data, gen_test_label = generate_data_maskd(test_loader, model, generate_lr=generate_lr,
                                                            generate_epochs=generate_epochs, mask=mask, init=init)

    gen_train_data = gen_train_data.numpy().reshape((len(gen_train_data), -1))
    gen_test_data = gen_test_data.numpy().reshape((len(gen_test_data), -1))
    gen_train_label = gen_train_label.numpy()
    gen_test_label = gen_test_label.numpy()

    other_train_data = other_set.data.numpy().reshape((len(other_set.data), -1))
    other_train_label = other_set.label.numpy()

    train_mad_ls, test_mad_ls = cal_mad(gen_train_data, gen_train_label,
                                        gen_test_data, gen_test_label,
                                        other_train_data, other_train_label,
                                        classes)
    if black_box:
        gen_shadow_train_data, gen_shadow_train_label = generate_data_black_box(train_loader, shadow_model,
                                                                                generate_lr=generate_lr,
                                                                                generate_epochs=generate_epochs,
                                                                                mask=mask, init=init
                                                                                )
        gen_shadow_test_data, gen_shadow_test_label = generate_data_black_box(test_loader, shadow_model,
                                                                              generate_lr=generate_lr,
                                                                              generate_epochs=generate_epochs,
                                                                              mask=mask, init=init
                                                                              )
    else:
        gen_shadow_train_data, gen_shadow_train_label = generate_data_maskd(train_loader, shadow_model,
                                                                            generate_lr=generate_lr,
                                                                            generate_epochs=generate_epochs, mask=mask,
                                                                            init=init
                                                                            )
        gen_shadow_test_data, gen_shadow_test_label = generate_data_maskd(test_loader, shadow_model,
                                                                          generate_lr=generate_lr,
                                                                          generate_epochs=generate_epochs, mask=mask,
                                                                          init=init
                                                                          )

    gen_shadow_train_data = gen_shadow_train_data.numpy()
    gen_shadow_test_data = gen_shadow_test_data.numpy()

    gen_shadow_train_label = gen_shadow_train_label.numpy()
    gen_shadow_test_label = gen_shadow_test_label.numpy()

    # reshape
    gen_shadow_train_data = gen_shadow_train_data.reshape((len(gen_shadow_train_data), -1))
    gen_shadow_test_data = gen_shadow_test_data.reshape((len(gen_shadow_test_data), -1))

    shadow_train_mad_ls, shadow_test_mad_ls = cal_mad(gen_shadow_train_data, gen_shadow_train_label,
                                                      gen_shadow_test_data, gen_shadow_test_label, other_train_data,
                                                      other_train_label, classes)

    assert len(train_mad_ls) == len(test_mad_ls), "train,test length not equal"
    assert len(shadow_train_mad_ls) == len(shadow_test_mad_ls), "shadow_train,shadow_test length not equal"

    # 去掉nan
    train_nan_mask = np.bitwise_or(np.isnan(train_mad_ls), np.isnan(shadow_train_mad_ls))
    test_nan_mask = np.bitwise_or(np.isnan(test_mad_ls), np.isnan(shadow_test_mad_ls))
    train_mad_ls, shadow_train_mad_ls = train_mad_ls[~train_nan_mask], shadow_train_mad_ls[~train_nan_mask]
    test_mad_ls, shadow_test_mad_ls = test_mad_ls[~test_nan_mask], shadow_test_mad_ls[~test_nan_mask]

    train_mad_ls, shadow_train_mad_ls = train_mad_ls[:len(test_mad_ls)], shadow_train_mad_ls[:len(test_mad_ls)]
    test_mad_ls, shadow_test_mad_ls = test_mad_ls[:len(train_mad_ls)], shadow_test_mad_ls[:len(train_mad_ls)]

    effective_len = len(train_mad_ls)
    print(f"有效的样本数量：{effective_len}")

    # 计算分数
    train_score = shadow_train_mad_ls / (train_mad_ls + 1e-8)
    test_score = shadow_test_mad_ls / (test_mad_ls + 1e-8)

    fpr, tpr, thresholds = roc_curve(
        np.array([1] * len(train_score) + [0] * len(test_score)),
        np.concatenate([train_score, test_score])
    )
    return fpr, tpr, thresholds


def attack_upgraded(dataset, model_name, generate_epochs=1, generate_lr=100, known_percentage="loo", grad_norm=False):
    inf_model = get_architecture("NCI_" + dataset).to(device)
    inf_model, _ = load_model(inf_model, f"../checkpoints/{dataset}_NCI.pt")
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    shadow_model = get_architecture(model_name + '_' + dataset).to(device)
    shadow_model, _ = load_model(shadow_model, f"../checkpoints/{dataset}_{model_name}_shadow.pt")

    train_set, test_set, other_set = load_dataset(dataset, ["member", "non_member", "aux0"])
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 创建结果存储文件夹
    outcome_dir = f"../outcome/performance_eval/ours/{dataset}_{model_name}"
    os.makedirs(outcome_dir, exist_ok=True)

    # for known_percentage in range(10, 101, 10):
    # for known_percentage in list(range(10, 99, 10)) + ['loo']:  # 10,20,30,...,90,leave-one-out
    if known_percentage == 'loo':
        known_feature_num = f_num - 1
        unknown_feature_num = 1
    else:
        known_feature_num = int(f_num * (known_percentage / 100))
        unknown_feature_num = f_num - known_feature_num

    print(f"known_percentage: {known_feature_num}/{f_num}")

    if mask is None:
        mask = generate_mask(mask_shape, unknown_num=unknown_feature_num)
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    # 生成攻击测试样本
    gen_train_data, gen_train_label = generate_data_maskd(train_loader, model, generate_lr=generate_lr,
                                                          generate_epochs=generate_epochs, mask=mask,
                                                          grad_norm=grad_norm)
    gen_test_data, gen_test_label = generate_data_maskd(test_loader, model, generate_lr=generate_lr,
                                                        generate_epochs=generate_epochs, mask=mask, grad_norm=grad_norm)

    train_pred_vec = get_pred_vec(shadow_model, gen_train_data.cuda()).cpu().numpy()
    test_pred_vec = get_pred_vec(shadow_model, gen_test_data.cuda()).cpu().numpy()
    other_pred_vec = get_pred_vec(shadow_model, other_set.data.contiguous().cuda()).cpu().numpy()

    gen_train_label = gen_train_label.numpy()
    gen_test_label = gen_test_label.numpy()

    other_train_label = other_set.label.numpy()

    gen_shadow_train_data, gen_shadow_train_label = generate_data_maskd(train_loader, shadow_model,
                                                                        generate_lr=generate_lr,
                                                                        generate_epochs=generate_epochs, mask=mask,
                                                                        grad_norm=grad_norm)
    gen_shadow_test_data, gen_shadow_test_label = generate_data_maskd(test_loader, shadow_model,
                                                                      generate_lr=generate_lr,
                                                                      generate_epochs=generate_epochs, mask=mask,
                                                                      grad_norm=grad_norm)

    shadow_train_pred_vec = get_pred_vec(inf_model, gen_shadow_train_data.cuda()).cpu().numpy()
    shadow_test_pred_vec = get_pred_vec(inf_model, gen_shadow_test_data.cuda()).cpu().numpy()
    shadow_other_pred_vec = get_pred_vec(inf_model, other_set.data.contiguous().cuda()).cpu().numpy()

    gen_shadow_train_label = gen_shadow_train_label.numpy()
    gen_shadow_test_label = gen_shadow_test_label.numpy()

    train_mad_ls, test_mad_ls = cal_mad(train_pred_vec, gen_train_label, test_pred_vec, gen_test_label,
                                        other_pred_vec, other_train_label, classes)

    shadow_train_mad_ls, shadow_test_mad_ls = cal_mad(shadow_train_pred_vec, gen_shadow_train_label,
                                                      shadow_test_pred_vec, gen_shadow_test_label,
                                                      shadow_other_pred_vec, other_train_label, classes)

    effective_len = len(train_mad_ls)
    print(f"有效的样本数量：{effective_len}")

    train_score = test_mad_ls
    test_score = train_mad_ls
    fpr, tpr, thresholds = roc_curve(np.array([1] * len(test_score) + [0] * len(train_score)),
                                     np.concatenate([test_score, train_score]))
    return fpr, tpr, thresholds
