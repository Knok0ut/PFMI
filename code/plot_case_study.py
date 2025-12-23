import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 方法列表与绘图样式
method_list = ["PMI", "CADE", "MSAD", "NCI"]
method_colors = ["blue", "green", "red", "purple"]
method_styles = ['-', '--', '-.', ':']

# 数据文件路径
base_path = "../outcome/case_study"

# 初始化图
plt.figure(figsize=(8, 6))

# 保存所有方法的数据用于CSV输出
roc_data = {}

for i, method in enumerate(method_list):
    file_path = os.path.join(base_path, f"{method}.npz")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # 加载数据
    data = np.load(file_path)
    fpr, tpr = data["fpr"], data["tpr"]

    # 排序（确保 ROC 曲线正常绘制）
    sorted_idx = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_idx]
    tpr = np.array(tpr)[sorted_idx]

    # 绘制
    plt.plot(fpr, tpr, label=method,
             color=method_colors[i],
             linestyle=method_styles[i],
             marker='o', markersize=4)

    # 存入字典，稍后导出CSV
    roc_data[f"{method}_fpr"] = fpr
    roc_data[f"{method}_tpr"] = tpr

# 设置对数坐标
plt.xscale("log")
plt.yscale("log")
plt.xlabel("False Positive Rate (log scale)", fontsize=14)
plt.ylabel("True Positive Rate (log scale)", fontsize=14)
plt.title("ROC Curve Comparison", fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# 保存图像
os.makedirs("../outcome/case_study/img", exist_ok=True)
plt.savefig("../outcome/case_study/img/roc_comparison.png", dpi=300)
plt.close()

# === 导出 CSV ===
# 补齐不同方法长度，用 NaN 填充对齐列数
max_len = max(len(v) for v in roc_data.values())
for key in roc_data:
    roc_data[key] = np.pad(roc_data[key], (0, max_len - len(roc_data[key])), constant_values=np.nan)

df = pd.DataFrame(roc_data)
os.makedirs("../outcome/case_study/csv", exist_ok=True)
df.to_csv("../outcome/case_study/csv/roc_comparison.csv", index=False)

print("图像和CSV数据已保存完成。")


