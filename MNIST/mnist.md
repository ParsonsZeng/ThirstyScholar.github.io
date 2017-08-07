# Classifying MNIST with Feedforward, Recurrent and Convolutional Neural Networks

***

Code structure: 

1. 前面都一樣
2. Define 3 types of networks using PyTorch.


2. 到 mini-batch sampling 時利用取出的 mini-batch 同時訓練三種 network。
3. 最後展示三種模型的 training/test accuracy。

***

This will be a post on classifying the MNIST dataset with different architectures of neural network.



*Architecture.*

1. **Feedforward Neural Network (FNN)**

   ​

   ![fnn](fnn.png)

   ​

   [details goes here...]

   ​


2. **Recurrent Neural Network (RNN)**

   A special kind of RNN we are considering here is the *LSTM* networks.

   ​

   ![lstm](lstm.png)

   ​

   [details goes here...]

   ​


3. **Convolutional Neural Network (CNN)**

   ​

   ![cnn](cnn.png)

   ​

   [details goes here...]

   ​



We implement the networks and training algorithm using `PyTorch`, Facebook's deep learning library. The same training procedure will be carried out for each neural network and training conditions will be fixed across each model for a meaningful comparison (although the model complexity, i.e. number of parameter in a model, is very different).

The other point we will stress in this post is *data preprocessing* which is the crucial part of the training procedure. Standard data preprocessing procedure for image data is illustrated below (this part is taken from the Stanford CS231n [class note](http://cs231n.github.io/neural-networks-2/)).



*Data Preprocessing.*

1. **Mean Substraction.**

   ​

2. **Normalization.**

   ​

*Code.*

[important code chunk goes here...]