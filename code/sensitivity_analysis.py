import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--tex", action="store_true")
args = parser.parse_args()

file_root = "../outcome/parameter_sensitivity/"
datasets = ["cifar10", "stl10", "fashion", "epsilon"]


def get_res(data):
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_diff = data_max - data_min
    data_diff = data_diff / 2
    data_mean = data.mean(axis=0)
    return data_mean, data_diff


methods = [
    'ours',
    'MSAD',
    'NCI',
    'CADE',
]

for method in methods:
    for dataset in datasets:
        aucs = []
        tprs = []
        for exp_idx in range(10):
            auc_path = f"sensitivity_{dataset}_{exp_idx}_{method}_auc.npy"
            tpr_path = f"sensitivity_{dataset}_{exp_idx}_{method}_tpr.npy"
            auc = np.load(os.path.join(file_root, auc_path))
            tpr = np.load(os.path.join(file_root, tpr_path))
            aucs.append(auc)
            tprs.append(tpr)

        aucs = np.stack(aucs)
        tprs = np.stack(tprs)
        auc_mean, auc_diff = get_res(aucs)
        tpr_mean, tpr_diff = get_res(tprs)
        np.savetxt(os.path.join(file_root, f"{dataset}_{method}_auc_mean.csv"), auc_mean, "%f", delimiter=",")
        np.savetxt(os.path.join(file_root, f"{dataset}_{method}_auc_diff.csv"), auc_diff, "%f", delimiter=",")
        np.savetxt(os.path.join(file_root, f"{dataset}_{method}_tpr_mean.csv"), tpr_mean, "%f", delimiter=",")
        np.savetxt(os.path.join(file_root, f"{dataset}_{method}_tpr_diff.csv"), tpr_diff, "%f", delimiter=",")
