#!/bin/bash

backbones=("resnet50" "resnet50" "mlp" "resnet50")
datasets=("cifar10" "fashion" "epsilon" "stl10")
lrs=("0.01" "0.01" "0.1" "0.01")
batch_sizes=(256 256 256 64)
num_labels=(10 10 2 10)

# 遍历每个实验设定
# shellcheck disable=SC2068
for i in ${!datasets[@]}; do
    backbone=${backbones[$i]}
    dataset=${datasets[$i]}
    labels=${num_labels[$i]}
    lr=${lrs[$i]}
    batch_size=${batch_sizes[$i]}

    echo "Running experiment: dataset=$dataset, backbone=$backbone"

    # 遍历每个label
    for ((label=0; label<$labels; label++)); do
        echo "Running label $label"
        python MSAD_main.py --backbone $backbone --dataset $dataset --label $label --lr $lr --batch_size $batch_size --epochs 500
    done
done