import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable


class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_as, self.ep_rs = [], []

        self.model = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.Tanh(),
            nn.Linear(10, self.n_actions),
            nn.Softmax()
        )

        # # Print model
        # print(self.model)
        # for param in self.model.parameters(): print(param)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, obs):
        s = Variable(torch.from_numpy(obs.astype(np.float32))).unsqueeze(0)
        prob_weights = self.model.forward(s)

        prob_weights_numpy = prob_weights.data.numpy()
        action = np.random.choice(range(prob_weights_numpy.shape[1]), p=prob_weights_numpy[0])

        _one_hot = np.zeros((1, 2))
        _one_hot[0][action] = 1
        one_hot = Variable(torch.from_numpy(_one_hot.astype(np.float32)))

        # Append log prob of the action taken
        self.ep_as.append(torch.log(torch.sum(prob_weights * one_hot)))

        return action

    def store_reward(self, r):
        self.ep_rs.append(r)

    def learn(self):
        # Compute return
        vt = self._compute_vt()

        # Learning
        loss = Variable(torch.zeros(1, 1))
        for i in range(len(self.ep_as)): loss += float(vt[i]) * self.ep_as[i]

        self.optimizer.zero_grad()
        (-loss).backward(retain_variables=True)
        self.optimizer.step()

        # Empty action/reward list after an episode of learning
        self.ep_as, self.ep_rs = [], []

    def _compute_vt(self):
        vt = np.zeros_like(self.ep_rs)

        running_sum = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_sum = running_sum * self.gamma + self.ep_rs[t]  # sum of discounted rewards
            vt[t] = running_sum

        '''
        Meaning of z scoring scaling:
            1. Value magnitude
            2. Encourage actions from the previous half -> actions preventing the pole from falling
               Discourage actions from the latter half -> actions causing the pole to fall
        '''
        vt = preprocessing.scale(vt)

        return vt
