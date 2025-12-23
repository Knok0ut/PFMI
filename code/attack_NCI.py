import os
from util.NCI import *
from util.attacks import NCI_attack
import argparse
import warnings
import torch

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_attack(dataset, model_name, generate_epochs=1, generate_lr=100, black_box=False):
    # 创建结果存储文件夹
    outcome_dir = f"../outcome/performance_eval/NCI/{dataset}_{model_name}"
    os.makedirs(outcome_dir, exist_ok=True)

    # for known_percentage in range(10, 101, 10):
    for known_percentage in list(range(10, 99, 10)) + ['loo']:  # 10,20,30,...,90,leave-one-out
        fpr, tpr, thresholds = NCI_attack(dataset, model_name, generate_epochs, generate_lr,
                                          known_percentage=known_percentage,
                                          black_box=black_box)
        # 保存fpr、tpr
        if black_box:
            save_path = os.path.join(outcome_dir, f"known_{known_percentage}_black.npz")
        else:
            save_path = os.path.join(outcome_dir, f"known_{known_percentage}.npz")
        np.savez(save_path, fpr=fpr, tpr=tpr)
        print(f"Saved FPR/TPR to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--black_box", action="store_true")
    parser.add_argument("--generate_lr", type=float, default=1000)
    args = parser.parse_args()
    setting_list = [
        ("cifar10", "resnet50"),
        ("fashion", "resnet50"),
        ("epsilon", "mlp"),
        ("stl10", "resnet50")
    ]
    for dataset, model_name in setting_list:
        run_attack(dataset, model_name, black_box=args.black_box, generate_lr=args.generate_lr)
