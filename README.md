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
Angular softmax loss feature map in MNIST.
![train features](https://github.com/Joyako/SphereFace-pytorch/blob/master/data/train/train_features.gif)
![test features](https://github.com/Joyako/SphereFace-pytorch/blob/master/data/test/test_features.gif)


## Formula
1.The original softmax loss is defined as:  
![Original Softmax Loss](https://github.com/Joyako/SphereFace-pytorch/blob/master/data/formalu/Screen%20Shot%202019-03-31%20at%2011.29.07%20AM.png)

2.The L-Softmax loss is defined as:  
![L-Softmax Loss](https://github.com/Joyako/SphereFace-pytorch/blob/master/data/formalu/Screen%20Shot%202019-03-31%20at%2011.46.53%20AM.png)

3.Modified softmax loss(Normalization: ||w|| = 1, bais = 0):  
![Modified softmax loss](https://github.com/Joyako/SphereFace-pytorch/blob/master/data/formalu/Screen%20Shot%202019-03-31%20at%2011.51.41%20AM.png)

4.A-Softmax loss:  
![A-Softmax loss](https://github.com/Joyako/SphereFace-pytorch/blob/master/data/formalu/Screen%20Shot%202019-03-31%20at%2011.55.58%20AM.png)

