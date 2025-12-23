python NCI_train.py --dataset cifar10 --epochs 150 --batch_size 256 --lr 0.01 $1
python NCI_train.py --dataset stl10 --epochs 150 --batch_size 64 --lr 0.01 $1
python NCI_train.py --dataset fashion --epochs 150 --batch_size 256 --lr 0.01 $1
python NCI_train.py --dataset epsilon --epochs 150 --batch_size 256 --lr 0.1 $1