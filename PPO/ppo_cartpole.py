from running_stat import ZFilter

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import gym

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

plt.style.use('seaborn-paper')


class PPO_Actor(nn.Module):
    def __init__(self, n_obs, hidden_units, n_act):
        super(PPO_Actor, self).__init__()
        self.hidden = nn.Linear(n_obs, hidden_units)
        self.action = nn.Linear(hidden_units, n_act)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        a = F.softmax(self.action(x))
        return a


class PPO_Critic(nn.Module):
    def __init__(self, n_obs, hidden_units):
        super(PPO_Critic, self).__init__()
        self.hidden = nn.Linear(n_obs, hidden_units)
        self.value = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        v = self.value(x)
        return v


env = gym.make('CartPole-v0')

NUM_EP = 500
NUM_EPOCH = 5
GAMMA = .99   # discount factor
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

EPS = .2
ACTOR_LR = 2.5e-4
CRITIC_LR = 5e-4
HIDDEN_SIZE = 128

zfilter = ZFilter(N_S)

actor = PPO_Actor(N_S, HIDDEN_SIZE, N_A)
actor_old = deepcopy(actor)

actor_optim = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)

critic = PPO_Critic(N_S, HIDDEN_SIZE)

critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
huber_loss = nn.SmoothL1Loss(size_average=True)


_Var = lambda x, dtype: Variable(torch.from_numpy(x).type(dtype))

# Tracker dict
traj = { 's': [], 'a': [], 'r': [], 'T': [] }

ep_ret = []
for i_episode in range(NUM_EP):
    s = env.reset()

    ret = 0
    while True:
        # Show last 10 episodes
        # if i_episode > NUM_EP - 10: env.render()

        s = zfilter(s)
        S = _Var(s, torch.FloatTensor).unsqueeze(0)
        A = actor.forward(S)

        a_prob = A.data.numpy()[0]
        a = int(np.random.choice(a_prob.shape[0], p=a_prob))

        s_, r, done, info = env.step(a)
        ret += r

        # Modify the reward function
        r = 0 if not done else -1

        traj['s'].append(s)
        traj['a'].append(np.array([a]))
        traj['r'].append(np.array([r]))
        traj['T'].append(not done)

        # Swap states
        s = s_

        # Update when collected a mini-batch of size 32
        if len(traj['s']) == 32:
            np_s = np.array(traj['s'])
            np_a = np.array(traj['a'])
            np_r = np.array(traj['r'])

            # Swap old actor network
            actor_old.load_state_dict(actor.state_dict())

            S_ = _Var(s_, torch.FloatTensor).unsqueeze(0)
            V_ = critic.forward(S_).data.numpy()[0][0]

            # Compute ^R_t (estimator for Q(s_t, a_t))
            n_step = []
            _ret = V_
            for _r, T in zip(reversed(traj['r']), traj['T']):
                _ret = _r + GAMMA * _ret * T   # truncate the return if ep terminates
                n_step.append(_ret)

            n_step.reverse()   # in-place reverse
            n_step = np.array(n_step)

            S = _Var(np_s, torch.FloatTensor)
            v = critic.forward(S).data.numpy()

            N_step = _Var(n_step, torch.FloatTensor)
            Adv = _Var(n_step - v, torch.FloatTensor)

            Act = _Var(np_a, torch.LongTensor)
            A_old = actor_old.forward(S).detach().gather(1, Act)

            for i in range(NUM_EPOCH):
                V = critic.forward(S)
                A_new = actor.forward(S).gather(1, Act)

                # Optimize critic
                critic_loss = huber_loss(V, N_step)
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                # Construct surrogate objective with epsilon 0.2
                ratio = A_new / A_old * Adv
                clamp_ratio = torch.clamp(A_new / A_old, min=1 - EPS, max=1 + EPS) * Adv
                surr_obj = torch.min(ratio, clamp_ratio)

                # Optimize actor
                actor_loss = -surr_obj.mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

            for k in traj.keys(): traj[k] = []

        if done:
            ep_ret.append(ret)
            print(ret)
            break

plt.title('Learning Curve')
plt.plot(ep_ret)
plt.show()
