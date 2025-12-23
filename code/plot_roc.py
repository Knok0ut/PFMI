import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

# 数据集和模型设定
setting_list = [
    ("cifar10", "resnet50"),
    ("fashion", "resnet50"),
    ("epsilon", "mlp"),
    ("stl10", "resnet50")
]

method_list = ["ours", "CADE", "MSAD", "NCI"]
method_colors = ["blue", "green", "red", "purple"]
method_styles = ['-', '--', '-.', ':']

# 输出路径
output_dir = "../outcome/performance_eval/roc_curves"
os.makedirs(output_dir, exist_ok=True)

# 已知特征比例设定
known_percentages = list(range(50, 100, 10))  # 50, 60, 70, 80, 90

for dataset, model_name in setting_list:
    for known_percentage in known_percentages:
        plt.figure(figsize=(7, 6))
        for idx, method in enumerate(method_list):
            base_dir = f"../outcome/performance_eval/{method}"

            file_path = os.path.join(base_dir, f"{dataset}_{model_name}", f"known_{known_percentage}.npz")

            if not os.path.exists(file_path):
                print(f"[Warning] {file_path} not found, skipping.")
                continue

            data = np.load(file_path)
            fpr = data['fpr']
            tpr = data['tpr']

            plt.plot(fpr, tpr,
                     label=method,
                     linestyle=method_styles[idx],
                     color=method_colors[idx],
                     linewidth=2)

        # 图设置
        plt.xlabel("False Positive Rate (FPR)", fontsize=13)
        plt.ylabel("True Positive Rate (TPR)", fontsize=13)
        plt.xscale('log')
        plt.title(f"{dataset.upper()} - ROC @ {known_percentage}%", fontsize=15)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        out_name = f"roc_{dataset}_{known_percentage}.png"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300)
        plt.close()

        print(f"[Saved] {out_name}")

import csv

csv_output_dir = "../outcome/performance_eval/roc_csv/"
os.makedirs(csv_output_dir, exist_ok=True)

for dataset, model_name in setting_list:
    for known_percentage in known_percentages:
        method_data = {}

        for idx, method in enumerate(method_list):

            base_dir = f"../outcome/performance_eval/{method}"

            file_path = os.path.join(base_dir, f"{dataset}_{model_name}", f"known_{known_percentage}.npz")

            if not os.path.exists(file_path):
                print(f"[Warning] {file_path} not found for CSV, skipping.")
                continue

            data = np.load(file_path)
            fpr = data['fpr']
            tpr = data['tpr']
            method_data[method] = (fpr, tpr)

        # === 写入 CSV ===
        csv_file = os.path.join(csv_output_dir, f"roc_{dataset}_{known_percentage}.csv")
        all_lengths = [len(v[0]) for v in method_data.values()]
        max_len = max(all_lengths) if all_lengths else 0

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = []
            for method in method_list:
                if method in method_data:
                    header.extend([f"FPR_{method}", f"TPR_{method}"])
            writer.writerow(header)

            # Data rows
            for i in range(max_len):
                row = []
                for method in method_list:
                    if method in method_data:
                        fpr, tpr = method_data[method]
                        if i < len(fpr):
                            row.extend([fpr[i], tpr[i]])
                        else:
                            row.extend(["", ""])
                writer.writerow(row)

        print(f"[CSV Saved] {csv_file}")
