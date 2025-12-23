# code execution
check code/requirements.txt for environment requirements
> mkdir checkpoints data outcome
>
>cd code/
### 1. split dataset
datasets: cifar10,stl10,fashion,epsilon

epsilon is available at [this url](https://storage.mds.yandex.net/get-devtools-opensource/250854/epsilon.tar.gz)
 ,md5:5bbfac403ac673da7d7ee84bd532e973

extract the gz file and put train.csv at ../data/epsilon/

> python split_dataset.py --dataset "dataset_name"

### 2. train target/shadow model
> python train.py --model "model_name" --dataset "dataset_name" --epoch 150 --lr 0.001
> 
> python train.py --model "model_name" --dataset "dataset_name" --epoch 150 --lr 0.001 --shadow


for example:
> python train.py --model resnet50  --dataset cifar10 --epoch 150 --lr 0.01 --batch_size 256 
> 
> python train.py --model resnet50 --dataset cifar10 --epoch 150 --lr 0.01 --batch_size 256  --shadow
>
> python train.py --model resnet50 --dataset stl10 --epoch 150 --lr 0.01 --batch_size 64 
> 
> python train.py --model resnet50 --dataset stl10 --epoch 150 --lr 0.01 --batch_size 64  --shadow
> 
> python train.py --model resnet50 --dataset fashion --epoch 150 --lr 0.01 --batch_size 256 
> 
> python train.py --model resnet50 --dataset fashion --epoch 150 --lr 0.01 --batch_size 256  --shadow
> 
> python train.py --model mlp --dataset epsilon --epoch 150 --lr 0.1 --batch_size 256 
> 
> python train.py --model mlp --dataset epsilon --epoch 150 --lr 0.1 --batch_size 256  --shadow

### 3. Prepare for CADE/NCI/MSAD
> python CADE_train.py --dataset "dataset_name"
> 
> ./NCI_train.sh
> 
> ./MSAD_train.sh

### 4. V-B. Performance Evaluation
> python attack_ours.py
> 
> python attack_CADE.py
> 
> python attack_NCI.py
> 
> python attack_MSAD.py

get raw data of plot:
> python to_plot.py --metric "auc/tpr@0.1"

### 5. V-C Hyperparameter Sensitivity
> python parameter_sensitivity.py --param_type "lr/epoch" --num_exp 10
> python sensitivity_analysis.py

### 6. V-D  Ablation Study
> python ablation_study.py


### 7. V-E. Impact of Feature Importance on Attack Performance
> ./get_shap_val.sh    # run this script for 4 datasets to get importance score
> 
> python shap_attack.py

### 7. V-F. black-box evaluation
> python attack_ours.py --black_box
> 
> python attack_CADE.py --black_box
> 
> python attack_NCI.py --black_box
> 
> python attack_MSAD.py --black_box