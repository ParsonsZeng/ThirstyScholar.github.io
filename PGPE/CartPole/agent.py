import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable


class PGPE:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features

        self.ita = 0.9
        self.b = 0.
        self.r = 0.
        self.R = []
        self.R.append(0.)

        self.model = nn.Sequential(
            nn.Linear(self.n_features, self.n_actions, bias=False),
            nn.Softmax()  # deterministic policy, pick action with greater value
        )

        self.a = Variable(torch.zeros(1, n_actions))

        self.Obs = []
        for _ in range(self.n_features): self.Obs.append([0.])

        self.Mu_lr = 0.2
        self.Sigma_lr = 0.1

        self.Param = list(self.model.parameters())
        self.Mu = []
        self.Sigma = []
        for p in self.Param:  # initialize hyper-params
            self.Mu.append(torch.zeros(p.size()))
            self.Sigma.append(torch.ones(p.size()) * 2)

            p.data = torch.normal(self.Mu[-1], self.Sigma[-1])
            # init.normal(p.data, mean=1, std=2)

    def choose_action(self, obs):
        temp = []
        for i in range(self.n_features):
            self.Obs[i].append(obs[i])
            temp.append(preprocessing.scale(np.array(self.Obs[i]))[-1])

        temp = np.array(temp)
        s = Variable(torch.from_numpy(temp.astype(np.float32))).unsqueeze(0)
        self.a = self.model.forward(s)
        a = self.a.data.numpy()
        action = a[0].argmax()
        # action = np.random.choice(np.arange(a.shape[1]), p=a[0])
        return action

    def get_reward(self):
        return self.r

    def store_return(self, r):
        self.r += r

    def learn_and_sample(self):
        # Process reward
        self.R.append(self.r / 200)

        # reset return tracker
        self.r = 0.

        # Freeze params if hit target reward
        if self.R[-1] >= 1.0: return

        # Learn and re-sample model parameters
        _r = float(preprocessing.scale(np.array(self.R))[-1])
        for i in range(len(self.Param)):
            # Learn
            _T = self.Param[i].data - self.Mu[i]
            _S = (_T**2 - self.Sigma[i]**2) / self.Sigma[i]

            _delta_Mu = self.Mu_lr * _r * _T
            self.Mu[i] += _delta_Mu

            _delta_Sigma = self.Sigma_lr * _r * _S
            self.Sigma[i] += _delta_Sigma

            # Re-sample
            self.Param[i].data = torch.normal(self.Mu[i], self.Sigma[i])
