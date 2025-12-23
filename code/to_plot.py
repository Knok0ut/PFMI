import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="auc")  # 支持 auc, tpr@0.1
parser.add_argument("--black_box", action="store_true")
args = parser.parse_args()

# 数据集设定
setting_list = [
    ("cifar10", "resnet50"),
    ("fashion", "resnet50"),
    ("epsilon", "mlp"),
    ("stl10", "resnet50")
]

# 方法设置
method_list = ["ours", "CADE", "MSAD", "NCI"]

# 输出路径
output_dir = "../outcome/performance_eval/img/"
os.makedirs(output_dir, exist_ok=True)
if args.black_box:
    suffix = "_black"
else:
    suffix = ""
for dataset, model_name in setting_list:
    known_percentages = list(range(10, 99, 10)) + ['loo']
    results_by_method = {method: [] for method in method_list}

    for known_percentage in known_percentages:
        for method in method_list:
            base_dir = f"../outcome/performance_eval/{method}"

            file_path = os.path.join(base_dir, f"{dataset}_{model_name}", f"known_{known_percentage}{suffix}.npz")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, using 0.0.")
                results_by_method[method].append(0.0)
                continue

            data = np.load(file_path)
            fpr = data['fpr']
            tpr = data['tpr']

            # 指标计算
            if args.metric == "auc":
                value = auc(fpr, tpr)
            elif args.metric.startswith("tpr@"):
                target_fpr = float(args.metric.split("@")[1])
                idxs = np.where(fpr <= target_fpr)[0]
                value = tpr[idxs[-1]] if len(idxs) > 0 else 0.0

            # print(f"{dataset}-{method}: {value}")
            else:
                raise ValueError(f"Unsupported metric: {args.metric}")

            results_by_method[method].append(value)

    # === 写入 CSV ===
    csv_path = os.path.join(output_dir, f"{dataset}_{args.metric.replace('@', '_')}_multicol{suffix}.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Known Percentage"] + method_list)

        for i, kp in enumerate(known_percentages):
            row = [kp] + [results_by_method[method][i] for method in method_list]
            writer.writerow(row)

    print(f"[Saved] {csv_path}")
