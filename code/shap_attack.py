import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from util.util import *
from util.datasets import load_dataset
from util.models import get_architecture, load_model
from util.attacks import get_MSAD_encoding as MSAD_encoding
from util.attacks import get_CADE_encoding as CADE_encoding
from util.MSAD import knn_score
from util.NCI import NCIPostprocessor

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
outcome_dir = "../outcome/shap"
os.makedirs(outcome_dir, exist_ok=True)


def unknown_mask_by_shap(shap, known_feature_num, set_large_unknow=False):
    """
    根据 SHAP 值为每个样本生成 unknown 特征掩码。

    参数:
        shap: Tensor, 形状为 (B, F) 或 (B, C, H, W)
        known_feature_num: int，每个样本中标记为已知的特征数量
        reverse: bool，为 True 表示将 SHAP 值小的标记为 False（unknown），
                      为 False 表示将 SHAP 值大的标记为 False（unknown）

    返回:
        mask: bool Tensor，形状为 (B, F) 或 (B, H, W)，True 表示已知，False 表示 unknown
    """
    if shap.ndim == 2:  # (B, F) - tabular
        B, F = shap.shape
        shap_flat = shap
        k = known_feature_num
        mask = torch.zeros_like(shap, dtype=torch.bool)

        topk_vals, topk_idx = torch.topk(shap_flat, k, dim=1, largest=set_large_unknow)

        row_idx = torch.arange(B).unsqueeze(1).expand(-1, k)
        mask[row_idx, topk_idx] = True
        return mask  # shape: (B, F)

    elif shap.ndim == 4:  # (B, C, H, W) - image
        B, C, H, W = shap.shape
        shap_mean = shap.mean(dim=1)  # (B, H, W)
        shap_flat = shap_mean.view(B, -1)  # (B, H*W)
        k = known_feature_num
        mask_flat = torch.zeros_like(shap_flat, dtype=torch.bool)

        topk_vals, topk_idx = torch.topk(shap_flat, k, dim=1, largest=set_large_unknow)

        row_idx = torch.arange(B).unsqueeze(1).expand(-1, k)
        mask_flat[row_idx, topk_idx] = True

        mask = mask_flat.view(B, H, W)
        return mask  # shape: (B, H, W)
    else:
        raise ValueError(f"Unsupported SHAP tensor shape: {shap.shape}")


def MRAD(dataset, model_name, train_loader, test_loader, generate_epochs=1,
         generate_lr=100):
    shadow_model = get_architecture(model_name + '_' + dataset).to(device)
    shadow_model, _ = load_model(shadow_model, f"../checkpoints/{dataset}_{model_name}_shadow.pt")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    other_set = load_dataset(dataset, ["aux0"])[0]
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 生成攻击样本
    gen_train_data, gen_train_label = generate_data_shap(train_loader, model, generate_lr=generate_lr,
                                                         generate_epochs=generate_epochs)
    gen_test_data, gen_test_label = generate_data_shap(test_loader, model, generate_lr=generate_lr,
                                                       generate_epochs=generate_epochs)

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

    gen_shadow_train_data, gen_shadow_train_label = generate_data_shap(train_loader, shadow_model,
                                                                       generate_lr=generate_lr,
                                                                       generate_epochs=generate_epochs,
                                                                       )
    gen_shadow_test_data, gen_shadow_test_label = generate_data_shap(test_loader, shadow_model,
                                                                     generate_lr=generate_lr,
                                                                     generate_epochs=generate_epochs,
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

    assert len(train_mad_ls) == len(test_mad_ls), f"train,test length not equal, {len(train_mad_ls)},{len(test_mad_ls)}"
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

    return fpr, tpr


def MSAD(dataset, model_name, train_loader, test_loader, generate_epochs=1, generate_lr=100):
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    other_set = load_dataset(dataset, ["aux0"])[0]
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 生成攻击样本
    gen_train_data, gen_train_label = generate_data_shap(train_loader, model, generate_lr=generate_lr,
                                                         generate_epochs=generate_epochs)
    gen_test_data, gen_test_label = generate_data_shap(test_loader, model, generate_lr=generate_lr,
                                                       generate_epochs=generate_epochs)

    mem_enc, mem_label = MSAD_encoding(model_name, dataset, gen_train_data, gen_train_label, numpy_flag=True)
    non_enc, non_label = MSAD_encoding(model_name, dataset, gen_test_data, gen_test_label, numpy_flag=True)
    other_enc, other_label = MSAD_encoding(model_name, dataset, other_set.data, other_set.label, numpy_flag=True)

    train_mad_ls = knn_score(other_enc, mem_enc)
    test_mad_ls = knn_score(other_enc, non_enc)

    fpr, tpr, thresholds = roc_curve(
        np.array([1] * len(train_mad_ls) + [0] * len(test_mad_ls)),
        np.concatenate([-train_mad_ls, -test_mad_ls])
    )
    return fpr, tpr


def NCI(dataset, model_name, train_loader, test_loader, generate_epochs=1, generate_lr=100):
    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    other_set = load_dataset(dataset, ["aux0"])[0]
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 生成攻击样本
    gen_train_data, gen_train_label = generate_data_shap(train_loader, model, generate_lr=generate_lr,
                                                         generate_epochs=generate_epochs)
    gen_test_data, gen_test_label = generate_data_shap(test_loader, model, generate_lr=generate_lr,
                                                       generate_epochs=generate_epochs)

    other_loader = DataLoader(other_set, batch_size=256, shuffle=False)
    shadow_model = get_architecture('NCI' + '_' + dataset).to(device)
    shadow_model, _ = load_model(shadow_model, f"../checkpoints/{dataset}_NCI.pt")
    processor = NCIPostprocessor(None)
    processor.setup(shadow_model, other_loader, None)
    _, train_dis = processor.postprocess(shadow_model, gen_train_data.cuda())
    _, test_dis = processor.postprocess(shadow_model, gen_test_data.cuda())

    fpr, tpr, thresholds = roc_curve(
        np.array([1] * len(train_dis) + [0] * len(test_dis)),
        np.concatenate([-train_dis.detach().cpu().numpy(), -test_dis.detach().cpu().numpy()])
    )
    return fpr, tpr


def CADE(dataset, model_name, train_loader, test_loader, generate_epochs=1, generate_lr=100):
    print(f"Running {dataset}-{model_name}...")

    model = get_architecture(model_name + '_' + dataset).to(device)
    model, _ = load_model(model, f"../checkpoints/{dataset}_{model_name}.pt")

    other_set = load_dataset(dataset, ["aux0"])[0]
    max_x, classes, mask_shape, f_num = get_ds_param(dataset)

    # 生成攻击样本
    gen_train_data, gen_train_label = generate_data_shap(train_loader, model, generate_lr=generate_lr,
                                                         generate_epochs=generate_epochs)
    gen_test_data, gen_test_label = generate_data_shap(test_loader, model, generate_lr=generate_lr,
                                                       generate_epochs=generate_epochs)

    AE = get_architecture(f"{dataset}_AE").to(device)
    AE, _ = load_model(AE, f"../checkpoints/AE/{dataset}_AE.pt")

    mem_set = TensorDataset(gen_train_data, gen_train_label)
    non_set = TensorDataset(gen_test_data, gen_test_label)
    mem_loader = DataLoader(mem_set, shuffle=False, batch_size=256)
    non_loader = DataLoader(non_set, shuffle=False, batch_size=256)
    other_loader = DataLoader(other_set, shuffle=False, batch_size=256)
    mem_enc, mem_label = CADE_encoding(AE, mem_loader, numpy_flag=True)
    non_enc, non_label = CADE_encoding(AE, non_loader, numpy_flag=True)
    other_enc, other_label = CADE_encoding(AE, other_loader, numpy_flag=True)
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
    return fpr, tpr


# Methods mapping to functions

method_funcs = {
    'ours': MRAD,
    'MSAD': MSAD,
    'NCI': NCI,
    'CADE': CADE,
}

if __name__ == '__main__':
    setting_list = [
        ('cifar10', 'resnet50'),
        ('fashion', 'resnet50'),
        ('epsilon', 'mlp'),
        ('stl10', 'resnet50'),
    ]
    known_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + ['loo']
    large_flag_list = [True, False]
    # known_percentages = [0.5]
    # large_flag_list = [False]

    summary_records = []

    for dataset, model_name in setting_list:
        max_x, classes, mask_shape, f_num = get_ds_param(dataset)

        train_pack = torch.load(f"../data/{dataset}/shap_train.pt")
        test_pack = torch.load(f"../data/{dataset}/shap_test.pt")

        train_data, train_label, train_shap = train_pack["datas"], train_pack["labels"], train_pack["shaps"]
        test_data, test_label, test_shap = test_pack["datas"], test_pack["labels"], test_pack["shaps"]

        if len(train_data.shape) == 4:  # image data
            bs, c, h, w, num_classes = train_shap.shape
            train_shap = train_shap.view(bs, c * h * w, num_classes)
            train_shap = train_shap[torch.arange(bs), :, train_label].view(bs, c, h, w)

            bs, c, h, w, num_classes = test_shap.shape
            test_shap = test_shap.view(bs, c * h * w, num_classes)
            test_shap = test_shap[torch.arange(bs), :, test_label].view(bs, c, h, w)
        elif len(train_data.shape) == 2:  # tabular data
            bs, fs, num_classes = train_shap.shape
            train_shap = train_shap[torch.arange(bs), :, train_label].view(bs, fs)

            bs, fs, num_classes = test_shap.shape
            test_shap = test_shap[torch.arange(bs), :, test_label].view(bs, fs)

        # Save AUC/TPR trend across known_percentage
        auc_trend = {f"{method}_K=IF": [] for method in method_funcs}
        auc_trend.update({f"{method}_K=UF": [] for method in method_funcs})
        tpr_trend = {f"{method}_K=IF": [] for method in method_funcs}
        tpr_trend.update({f"{method}_K=UF": [] for method in method_funcs})

        for kp in known_percentages:
            if kp == 'loo':
                unknown_num = f_num - 1
            else:
                unknown_num = int(f_num * (1 - kp))
            for large_flag in large_flag_list:
                train_mask = unknown_mask_by_shap(train_shap, unknown_num, set_large_unknow=large_flag)
                test_mask = unknown_mask_by_shap(test_shap, unknown_num, set_large_unknow=large_flag)

                train_loader = DataLoader(TensorDataset(train_data, train_label, train_mask), shuffle=False,
                                          batch_size=1024)
                test_loader = DataLoader(TensorDataset(test_data, test_label, test_mask), shuffle=False,
                                         batch_size=1024)

                for method, func in method_funcs.items():
                    fpr, tpr = func(dataset, model_name, train_loader, test_loader, generate_lr=1000)

                    auc_score = auc(fpr, tpr)
                    target_fpr = 0.1
                    idxs = np.where(fpr <= target_fpr)[0]
                    tpr_at_0_1fpr = tpr[idxs[-1]] if len(idxs) > 0 else 0.0

                    summary_records.append({
                        'dataset': dataset,
                        'model': model_name,
                        'method': method,
                        'known_percentage': kp,
                        'set_large_unknow': large_flag,
                        'auc': auc_score,
                        'tpr@0.1fpr': tpr_at_0_1fpr
                    })
                    key_suffix = f"{method}_{'K=UF' if large_flag else 'K=IF'}"
                    auc_trend[key_suffix].append(auc_score)
                    tpr_trend[key_suffix].append(tpr_at_0_1fpr)
        # Plot AUC / TPR trends
        for metric_name, metric_dict in zip(['auc', 'tpr@0.1fpr'], [auc_trend, tpr_trend]):
            plt.figure(figsize=(8, 6))
            color_map = {'ours': 'blue', 'MSAD': 'green', 'NCI': 'orange', 'CADE': 'red'}

            for method in method_funcs:
                for large_flag in large_flag_list:
                    key = f"{method}_{'K=UF' if large_flag else 'K=IF'}"
                    linestyle = '-' if large_flag else '--'
                    label = f"{method} ({'large' if large_flag else 'small'})"
                    plt.plot(known_percentages, metric_dict[key], linestyle=linestyle,
                             color=color_map[method], label=label)

            plt.xlabel('Known Feature Percentage')
            plt.ylabel(metric_name.upper())
            plt.title(f"{metric_name.upper()} Trend - {dataset}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(outcome_dir, f"{dataset}_{metric_name}_trend.png"))
            plt.close()
            pd_idx = [int(kp*100) for kp in known_percentages[:-1]]+['loo']
            pd.DataFrame(auc_trend, index=pd_idx).to_csv(
                os.path.join(outcome_dir, f"{dataset}_auc_trend.csv"))
            pd.DataFrame(tpr_trend, index=pd_idx).to_csv(
                os.path.join(outcome_dir, f"{dataset}_tpr_trend.csv"))

    # Save overall summary
    pd.DataFrame(summary_records).to_csv(os.path.join(outcome_dir, "summary_auc_tpr.csv"), index=False)
