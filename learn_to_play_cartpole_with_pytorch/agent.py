import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable


class PGPE:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features

        self.ret = 0.
        
        # List to store reward received from the env every step.
        # Append a zero first since at least two numbers is needed
        # to perform normalization(o.w. error will occur at the 
        # first step).
        self.R = []
        self.R.append(0.)

        self.model = nn.Sequential(
            nn.Linear(self.n_features, self.n_actions, bias=False),
            nn.Softmax()  # deterministic policy, pick action with greater value
        )

        # List to store observations(array of shape (1, 4))
        # Append a zero first for the same reason as above
        self.Obs = []
        for _ in range(self.n_features): self.Obs.append([0.])

        # Learning rate for hyper-params mu and sigma
        self.Mu_lr = 0.2
        self.Sigma_lr = 0.1

        # Prepare hyper-params, store mean/var separately in lists
        self.Param = list(self.model.parameters())
        self.Mu = []
        self.Sigma = []
        for p in self.Param:  
            # initialize hyper-params
            self.Mu.append(torch.zeros(p.size()))
            self.Sigma.append(torch.ones(p.size()))
            
            # Sample initial model params
            p.data = torch.normal(self.Mu[-1], self.Sigma[-1])

    def choose_action(self, obs):
        # Normalize observations
        temp = []
        for i in range(self.n_features):
            self.Obs[i].append(obs[i])
            temp.append(preprocessing.scale(np.array(self.Obs[i]))[-1])

        temp = np.array(temp)
        s = Variable(torch.from_numpy(temp.astype(np.float32))).unsqueeze(0)  # cast np array to torch variable
        a = self.model.forward(s).data.numpy()
        action = a[0].argmax()  # pick action with greater value
        return action

    def get_return(self):
        return self.ret

    def store_reward(self, r):
        # Compute undiscounted sum of rewards
        self.ret += r

    def learn_and_sample(self):
        # Scale reward to range [0, 1]
        # In the CartPole env, one condition for episode termination is
        # that agent cumulates +200 rewards. Env details can be found here:
        # https://github.com/openai/gym/wiki/CartPole-v0
        self.R.append(self.ret - 200)

        # reset return tracker
        self.ret = 0.

        # normalize reward signal
        _r = float(preprocessing.scale(np.array(self.R))[-1])

        # Learn and re-sample model parameters
        # This part is a direct implementation of the vanilla
        # version of the PGPE algorithm, original paper can be
        # found here:
        # http://kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Networks-2010-Sehnke_%5b0%5d.pdf
        # Left column of Algorithm 1 table on p.7 in the paper
        for i in range(len(self.Param)):
            # Learning
            # These are the T and S matrices in the original paper
            _T = self.Param[i].data - self.Mu[i]
            _S = (_T**2 - self.Sigma[i]**2) / self.Sigma[i]

            # Update means
            _delta_Mu = self.Mu_lr * _r * _T
            self.Mu[i] += _delta_Mu

            # Update standard deviations
            _delta_Sigma = self.Sigma_lr * _r * _S
            self.Sigma[i] += _delta_Sigma
            
            # Freeze params if hit target reward, else re-sample
            if self.R[-1] < 0.:
                self.Param[i].data = torch.normal(self.Mu[i], self.Sigma[i])
