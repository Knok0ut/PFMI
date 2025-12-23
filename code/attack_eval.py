import argparse
from sklearn.metrics import roc_curve, auc
from util.attacks import *
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
parser.add_argument("--generate_lr", type=float, default=1000)
parser.add_argument("--known_percentage", type=int, default=80)
parser.add_argument("--grad_norm", action="store_true")
parser.add_argument("--black_box", action="store_true")
parser.add_argument("--attack_type", type=str, default="NCI")
parser.add_argument("--_file_name", type=str, default=__file__)

args = parser.parse_args()

model_name = args.model
dataset = args.dataset
generate_epochs = args.generate_epochs
generate_lr = args.generate_lr
known_percentage = args.known_percentage
grad_norm = args.grad_norm
attack_type = args.attack_type
black_box = args.black_box

if attack_type == "MRAD":
    fpr, tpr, threshold = MRAD(dataset, model_name, generate_epochs, generate_lr, known_percentage)
elif attack_type == "NCI":
    fpr, tpr, threshold = NCI_attack(dataset, model_name, generate_epochs, generate_lr,
                                     known_percentage=known_percentage, black_box=black_box)
elif attack_type == "MSAD":
    fpr, tpr, threshold = MSAD_attack(dataset, model_name, generate_epochs, generate_lr, known_percentage,
                                      black_box=black_box)
elif attack_type == "CADE":
    fpr, tpr, threshold = CADE_attack(dataset, model_name, generate_epochs, generate_lr, known_percentage,
                                      black_box=black_box)
else:
    raise NotImplemented()

auc_score = auc(fpr, tpr)
print(f"attack: {attack_type}")
print(f"AUC: {float(auc_score)}")
