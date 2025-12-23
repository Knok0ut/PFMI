import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc

from util.attacks import *

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
outcome_dir = "../outcome/ablation_study"
os.makedirs(outcome_dir, exist_ok=True)

method_funcs = {
    'ours': MRAD,
    'CADE': CADE_attack,
    'MSAD': MSAD_attack,
    'NCI': NCI_attack
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Experiments')
    args = parser.parse_args()

    # Experimental settings: (dataset, model_name)
    setting_list = [
        ('cifar10', 'resnet50'),
        ('fashion', 'resnet50'),
        ('epsilon', 'mlp'),
        ('stl10', 'resnet50'),
    ]
    target_fpr = 0.1

    os.makedirs(outcome_dir, exist_ok=True)

    for dataset, model_name in setting_list:
        max_x, classes, mask_shape, f_num = get_ds_param(dataset)

        auc_rows = []
        tpr_rows = []
        kp_list = list(range(10, 99, 10)) + ['loo']

        for known_percentage in kp_list:
            if known_percentage == 'loo':
                known_feature_num = f_num - 1
                kp_val = 'loo'
            else:
                known_feature_num = int(f_num * (known_percentage / 100))
                kp_val = known_percentage

            unknown_feature_num = f_num - known_feature_num
            globals()['unknown_feature_num'] = unknown_feature_num

            auc_row = {'Known Percentage(%)': kp_val}
            tpr_row = {'Known Percentage(%)': kp_val}

            for method, func in method_funcs.items():
                try:
                    fpr, tpr, _ = func(dataset, model_name,
                                       generate_epochs=0,
                                       generate_lr=1000, init="random")
                    method_auc = auc(fpr, tpr)
                    idxs = np.where(fpr <= target_fpr)[0]
                    if len(idxs) > 0:
                        method_tpr = tpr[idxs[-1]]
                    else:
                        method_tpr = 0.0
                except Exception as e:
                    print(f"Error running {method} on {dataset}-{model_name} with kp={kp_val}: {e}")
                    method_auc, method_tpr = np.nan, np.nan

                auc_row[method] = method_auc
                tpr_row[method] = method_tpr

            auc_rows.append(auc_row)
            tpr_rows.append(tpr_row)

        # Save AUC CSV
        auc_df = pd.DataFrame(auc_rows)
        auc_path = os.path.join(outcome_dir, f'ablation_{dataset}_auc.csv')
        auc_df.to_csv(auc_path, index=False)

        # Save TPR CSV
        tpr_df = pd.DataFrame(tpr_rows)
        tpr_path = os.path.join(outcome_dir, f'ablation_{dataset}_tpr.csv')
        tpr_df.to_csv(tpr_path, index=False)

        print(f"Saved {dataset} AUC to {auc_path}")
        print(f"Saved {dataset} TPR@0.1FPR to {tpr_path}")
