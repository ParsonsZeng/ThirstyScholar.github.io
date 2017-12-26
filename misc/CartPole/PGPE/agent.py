import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class PGPE:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features

        self.ret = 0.

        self.model = nn.Sequential(
            nn.Linear(self.n_features, self.n_actions, bias=False),
            nn.Softmax()  # deterministic policy, pick action with greater value
        )

        # Learning rate for hyper-params mu and sigma
        self.Mu_lr = 0.2
        self.Sigma_lr = 0.1

        # Prepare hyper-params, store mean/var separately in lists
        self.Param = list(self.model.parameters())
        self.Mu = []
        self.Sigma = []
        for p in self.Param:
            # initialize hyper-params
            self.Mu.append(torch.normal(torch.zeros(p.size()), torch.ones(p.size())))
            self.Sigma.append(2 * torch.ones(p.size()))

            # Sample initial model params
            p.data = torch.normal(self.Mu[-1], self.Sigma[-1])

    def choose_action(self, obs):
        # Scale input to (-1, 1):
        #   1. if range is finite -> divide by range
        #   2. if range is infinite -> take tanh
        obs[0] /= 2.4
        obs[1] = np.tanh(obs[1])
        obs[2] /= 41.8
        obs[3] = np.tanh(obs[3])

        s = Variable(torch.from_numpy(obs.astype(np.float32))).unsqueeze(0)  # cast np array to torch variable
        a = self.model.forward(s).data.numpy()
        action = a[0].argmax()  # pick action with greater value
        return action

    def get_return(self):
        return self.ret

    def store_reward(self, r):
        self.ret += r  # Compute un-discounted sum of rewards

    def learn_and_sample(self):
        # Scale reward to [0, 1]
        _r = self.ret / 200

        # reset return tracker
        self.ret = 0.

        for i in range(len(self.Param)):
            # Learning
            # These are the T and S matrices in the original paper
            _T = self.Param[i].data - self.Mu[i]
            _S = (_T ** 2 - self.Sigma[i] ** 2) / self.Sigma[i]

            # Update means
            _delta_Mu = self.Mu_lr * _r * _T
            self.Mu[i] += _delta_Mu

            # Update standard deviations
            _delta_Sigma = self.Sigma_lr * _r * _S
            self.Sigma[i] += _delta_Sigma

            # Freeze params if hit target reward, else re-sample
            if _r < 1.:
                self.Param[i].data = torch.normal(self.Mu[i], self.Sigma[i])