from replay_memory import Transition, ReplayMemory
from running_stat import ZFilter

from itertools import zip_longest
from copy import deepcopy
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_index', help='args for file index', type=int)
args = parser.parse_args()

env = gym.make('Pendulum-v0')

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_BOUND = env.action_space.high

NUM_EP = 5

LR_A = 2.5e-4
LR_C = 5e-4
TAU = 1e-3
GAMMA = 0.99
BATCH_SIZE = 32
HIDDEN_SIZE = 128
CAPACITY = 2048


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.affine = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.affine(x))
        x = torch.clamp(self.output(x), min=-1, max=1)
        return x


class Critic(nn.Module):
    def __init__(self, s_size, a_size, hidden_size):
        super(Critic, self).__init__()
        self.affine_s = nn.Linear(s_size, hidden_size)
        self.affine_a = nn.Linear(a_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        x = F.relu(self.affine_s(s) + self.affine_a(a))
        x = self.output(x)
        return x


class DDPG(nn.Module):

    def __init__(self):
        super(DDPG, self).__init__()

        # Z filter
        self.zfilter = ZFilter(S_DIM)

        #
        self.actor = Actor(S_DIM, HIDDEN_SIZE, A_DIM)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), LR_A)

        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(S_DIM, A_DIM, HIDDEN_SIZE)
        self.critic_loss_fn = nn.SmoothL1Loss(size_average=True)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), LR_C)

        self.critic_target = deepcopy(self.critic)

        #
        self.replay_memory = ReplayMemory(capacity=CAPACITY)
        self._Var = lambda v, dtype: Variable(torch.from_numpy(v).type(dtype))

    def choose_action(self, s, noise):
        s = self.zfilter(s)
        S = self._Var(np.expand_dims(s, axis=0), torch.FloatTensor)

        action = self.actor(S).data.numpy()[0] * A_BOUND
        action += noise
        return action

    def store_transition(self, s, a, r, s_):
        s = self.zfilter(s)
        s_ = self.zfilter(s_)
        r /= 16   # scale reward signal

        self.replay_memory.push(s, a, np.array([r]), s_)   # store with 1D np array

    def learn(self):
        if len(self.replay_memory) < self.replay_memory.capacity: return

        transitions = self.replay_memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        S = self._Var(np.stack(batch.s), torch.FloatTensor)
        A = self._Var(np.stack(batch.a), torch.FloatTensor)
        R = self._Var(np.stack(batch.r), torch.FloatTensor)
        S_ = self._Var(np.stack(batch.s_), torch.FloatTensor)

        # Use both target network to compute TD-target
        Q_ = self.critic_target(S_, self.actor_target(S_)).detach()
        Q_target = R + GAMMA * Q_

        # Estimated Q-value
        Q_est = self.critic.forward(S, A)

        # Optimize critic
        C_loss = self.critic_loss_fn(Q_est, Q_target)
        self.critic_optim.zero_grad()
        C_loss.backward()
        self.critic_optim.step()

        # Optimize actor
        Q = self.critic.forward(S, self.actor.forward(S))

        A_loss = -Q.mean()
        self.actor_optim.zero_grad()
        A_loss.backward()
        self.actor_optim.step()

        # Soft update on target networks
        for c, c_t, a, a_t in zip_longest(self.critic.parameters(), self.critic_target.parameters(),
                                          self.actor.parameters(), self.actor_target.parameters()):
            if c is not None:
                c_t.data = TAU * c.data + (1 - TAU) * c_t.data
            if a is not None:
                a_t.data = TAU * a.data + (1 - TAU) * a_t.data


if __name__ == '__main__':
    ddpg = DDPG()

    ep_ret = []
    for ep in range(NUM_EP):
        s = env.reset()

        ret = 0
        while True:
            # # Render last 10 episodes
            # if ep > NUM_EP - 10: env.render()

            # Use a fixed noise from a normal distro.
            a = ddpg.choose_action(s, np.random.rand() * .1)
            s_, r, done, info = env.step(a)
            ret += r

            ddpg.store_transition(s, a, r, s_)
            s = s_

            ddpg.learn()

            if done:
                ep_ret.append(ret)
                print('Ep: ', ep, '| Ep_r: ', round(ret, 2))
                break

    import pickle
    i = 0 if args.file_index is None else args.file_index

    file_name = 'DDPG_Pendulum_{0}.pkl'.format(i)
    with open(file_name, mode='wb') as handle:
        pickle.dump(ep_ret, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved to {0}'.format(file_name))



