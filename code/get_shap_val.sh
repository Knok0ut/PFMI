#!/bin/bash
python SHAP.py --model resnet50 --dataset cifar10
python SHAP.py --model resnet50 --dataset fashion
python SHAP.py --model resnet50 --dataset stl10
python SHAP.py --model mlp --dataset epsilon
