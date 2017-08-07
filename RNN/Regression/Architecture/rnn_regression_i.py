import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# This shows good pic: format from classic style; color and style from seaborn
plt.style.use('classic')
plt.style.use('seaborn')

# Hyper-Parameters
TIME_STEP = 10  # mini-batch size
SKIP = 10  # skip some time-steps to prepare hidden state in mini-batch training

INPUT_SIZE = 1  # rnn input size
HIDDEN_SIZE = 16  # rnn hidden size
OUTPUT_SIZE = 1  # rnn output size

LR = 1e-2  # learning rate
MM = 0.5  # momentum
REG = 1e-4  # regularization strength (L2)


# ## Data

# Generate data: train a RNN to input cos(t) to predict sin(t)
# Note the np.float32 datatype for torch.FloatTensor
N = 100
steps = np.linspace(0, 10* np.pi, N, dtype=np.float32)[:, np.newaxis]  # time steps
X_np = np.cos(steps)
y_np = np.sin(steps)


# Plot data
plt.figure(1, figsize=(12, 5))
plt.plot(steps, X_np, 'b', linestyle='dashed', label='input: x = cos(t)')
plt.plot(steps, y_np, 'r', label='target: y = sin(t)')
plt.legend(loc='lower left')
plt.title('Data')
plt.show()


# ## Model: Network, Loss and Optimizer

class ElmanNet(nn.Module):
    """ Identical to the nn.RNN module in PyTorch """

    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanNet, self).__init__()

        self.FC1 = nn.Linear(input_size, hidden_size)
        self.RC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, last_hidden):
        hidden = F.relu(self.FC1(x) + self.RC2(last_hidden))
        output = self.FC3(hidden)

        return output, hidden

# Model
rnn = ElmanNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# Optimizer: try Momentum or Adam
# optim = torch.optim.SGD(rnn.parameters(), lr=LR, momentum=0.5, weight_decay=REG)
optim = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=REG)

# Loss function
loss_fn = nn.MSELoss()


"""
Before training, the RNN ouptuts the curve like this...
"""

plt.figure(1, figsize=(12, 5))

# Plot after all training completes
pred_lst = []
hidden = Variable(torch.zeros(1, HIDDEN_SIZE))  # initialize hidden state

for t in range(N):
    x = Variable(torch.from_numpy(X_np[t:t + 1, :]))

    predict, hidden = rnn.forward(x, hidden)
    pred_lst.append(predict.data[0][0])

plt.title('Before Training Prediction')
plt.plot(steps, y_np.flatten(), 'r', label='target')
plt.plot(steps, np.array(pred_lst), 'b', label='prediction')
plt.legend(loc='lower left')

plt.show()

# ## Training and Evaluation

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuous plotting, Jupyter notebook can't run this

# Train for __ epochs; an epoch means that every data point should be picked once in expectation
for epoch in range(50):
    plt.cla()

    # Adjust the number of iterations so that each data point is picked once for each epoch in expectation
    for iteration in range(int(N / TIME_STEP)):

        """
        *Trick* 
        It would be problematic to always start training with empty hidden state in each mini-batch, so we sample a 
        longer sequence (TIME_STEP + SKIP) and perform a complete forward pass; BUT only propagate for TIME_STEP many
        steps backward.
        """

        # Pick start index in the dataset to create a mini-batch
        start = np.random.randint(N)
        prestart = max(start - SKIP, 0)
        end = min(start + TIME_STEP, 200)

        trainX = X_np[prestart:end, :]
        trainY = y_np[prestart:end, :]

        # Initialize hidden state for each mini-batch
        hidden = Variable(torch.zeros(1, HIDDEN_SIZE))

        # Forward pass and loss compute
        Loss = 0
        for t in range(len(trainX)):  # note it may be less that TIME_STEP+SKIP many time-steps

            x = Variable(torch.from_numpy(trainX[t:t + 1, :]))
            y = Variable(torch.from_numpy(trainY[t:t + 1, :]))

            predict, hidden = rnn.forward(x, hidden)

            if t < start - prestart: continue  # skip the first SKIP many time-steps

            loss = loss_fn(predict, y)
            Loss += loss

        # Average loss
        Loss /= (end - start)

        # Backward pass and param update
        optim.zero_grad()
        Loss.backward()
        optim.step()


    """
    Training evaluation for each epoch
    """

    # Plot after all training completes
    pred_lst = []
    hidden = Variable(torch.zeros(1, HIDDEN_SIZE))  # initialize hidden state

    Loss = 0
    for t in range(N):
        x = Variable(torch.from_numpy(X_np[t:t + 1, :]))
        y = Variable(torch.from_numpy(y_np[t:t + 1, :]))

        predict, hidden = rnn.forward(x, hidden)

        pred_lst.append(predict.data[0][0])

        loss = loss_fn(predict, y)
        Loss += loss

    plt.title('Epoch {0} of Training'.format(epoch+1))
    plt.plot(steps, y_np.flatten(), 'r', label='target')
    plt.plot(steps, np.array(pred_lst), 'b', label='prediction')
    plt.legend(loc='lower left')
    plt.ylim((-1., 1.))
    plt.draw(); plt.pause(0.1)

plt.ioff()
plt.show()
