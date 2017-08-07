# Training a Recurrent Neural Network for a Regression Task

This is a series of posts on training recurrent neural networks (RNNs) to solve a simple regression task: predicting the value of $\sin(t)$ using the value of $\cos(t)$. The network is trained via standard backpropagation through time (BPTT). We attacked this problem from two perspectives:

1. Explore different *architectures* of RNN:

   - Elman network (h2h)
   - Jordan network (o2h)
   - *Hybrid network (= Elman + Jordan)
   - LSTM network

   All hyper-parameters are fixed across all networks to easily compare the results. The network and BPTT algorithm are implemented in `PyTorch`.

   *I made this name up, don't know if there is a specific name for this kind of network.

2. Implement the network and BPTT algorithm by hand using `Numpy`:

   Here we open up the block box and address the mechanism on training  RNNs. We implementing both the network and BPTT algorithm **from scratch.** The reader can easily understand the training mechanism by reading the code.

There is an *interactive* version for some of the code (ones with file name ended with `_i`). We visualize the training result for each epoch, which is fun to watch the network trying the fit the data. Unfortunately, the `matplotlib` continuous plotting functionality doesn't run on Jupyter notebook, so I provided the raw Python code and the reader can run the code elsewhere.