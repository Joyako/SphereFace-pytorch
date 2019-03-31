# SphereFace in Pytorch

**An implementation of [SphereFace:Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063).**
This project use [MNIST](https://github.com/Joyako/SphereFace-pytorch/tree/master/data/MNIST) as train data, which include 
network SphereFace4, SphereFace20 etc. and take shortcut connection implementation.

## How to use it, as follow:
1.Download
```bash
git clone git@github.com:Joyako/SphereFace-pytorch.git
```
    
2. Execution
```bash
python train.py
```

## Result
![Angular Softmax Loss. Left: train. Right: test.]()


## Formula
1.![Original Softmax Loss:](SphereFace-pytorch/data/formalu/Screen Shot 2019-03-31 at 11.29.07 AM.png)


    