import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc

from util.util import *
from util.datasets import load_dataset
from util.models import get_architecture, load_model
from util.attacks import CADE_attack, MRAD, MSAD_attack, NCI_attack

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
outcome_dir = "../outcome/parameter_sensitivity"
os.makedirs(outcome_dir, exist_ok=True)

# Sensitivity parameter ranges
generate_lr_list = [0.01, 0.1, 1, 10, 100, 1000, 10000]
generate_epochs_list = [1, 2, 5, 10, 20]

# Methods mapping to functions


method_funcs = {
    'ours': MRAD,
    'MSAD': MSAD_attack,
    'NCI': NCI_attack,
    'CADE': CADE_attack,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Experiments')
    parser.add_argument('--param_type', choices=['lr', 'epoch'], default='epoch',
                        help='Parameter to sweep: lr for generate_lr, epoch for generate_epochs')
    parser.add_argument('--known_percentage', type=float, default=80,
                        help='Known feature percentage (0-1) for mask generation')
    parser.add_argument('--num_exp', type=int, default=10,
                        help='Known feature percentage (0-1) for mask generation')
    args = parser.parse_args()
    known_percentage = args.known_percentage
    num_exp = args.num_exp

    # Choose parameter list based on param_type
    if args.param_type == 'lr':
        sweep_values = generate_lr_list
    else:
        sweep_values = generate_epochs_list

    # Experimental settings: (dataset, model_name)
    setting_list = [
        ('cifar10', 'resnet50'),
        ('fashion', 'resnet50'),
        ('epsilon', 'mlp'),
        ('stl10', 'resnet50'),
    ]

    # Prepare results container
    for exp_idx in range(num_exp):
        for dataset, model_name in setting_list:
            # for val in sweep_values:
            for method, func in method_funcs.items():
                auc_sheet = np.empty((len(generate_lr_list), len(generate_epochs_list)), dtype=float)
                tpr_sheet = np.empty((len(generate_lr_list), len(generate_epochs_list)), dtype=float)
                for lr_idx, gen_lr in enumerate(generate_lr_list):
                    for epo_idx, gen_epochs in enumerate(generate_epochs_list):
                        try:
                            fpr, tpr, thresholds = func(dataset, model_name,
                                                        generate_epochs=gen_epochs,
                                                        generate_lr=gen_lr, known_percentage=known_percentage)
                            target_fpr = 0.1
                            idxs = np.where(fpr <= target_fpr)[0]
                            if len(idxs) > 0:
                                tpr_at_0_1 = tpr[idxs[-1]]
                            else:
                                tpr_at_0_1 = 0.0
                            method_auc, method_tpr = auc(fpr, tpr), tpr_at_0_1
                        except Exception as e:
                            print(f"Error running {method} on {dataset}-{model_name} with {args.param_type}= {e}")
                            method_auc, method_tpr = np.nan, np.nan
                        auc_sheet[lr_idx, epo_idx] = float(method_auc)
                        tpr_sheet[lr_idx, epo_idx] = float(method_tpr)
            # Save results
                auc_path = os.path.join(outcome_dir, f'sensitivity_{dataset}_{exp_idx}_{method}_auc.npy')
                tpr_path = os.path.join(outcome_dir, f'sensitivity_{dataset}_{exp_idx}_{method}_tpr.npy')
                np.save(auc_path, np.asarray(auc_sheet))
                np.save(tpr_path, np.asarray(tpr_sheet))