from network import RecurrentA2C

import matplotlib.pyplot as plt
import numpy as np
import gym

import torch
from torch.autograd import Variable


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_index', help='args for file index', type=int)
args = parser.parse_args()


# Create environment
env = gym.make('CartPole-v0')
env.seed(1)  # reproducible

N_S = env.observation_space.shape[0]
N_A = env.action_space.n

agent = RecurrentA2C(N_S, N_A)

c = 1e-1
lr = 2.5e-4
optim = torch.optim.Adam(agent.parameters(), lr=lr)

BlurProb = .3
blur = lambda s, p: (1 - (np.random.rand(s.shape[0]) < p)).astype(np.float32) * s

#
Ep_r = []
for i_episode in range(600):
    s = blur(env.reset(), BlurProb)

    h_n = Variable(torch.zeros(1, 1, 128))
    c_n = Variable(torch.zeros(1, 1, 128))

    # Tracker lists
    s_lst, a_lst, r_lst = [], [], []

    ep_r = 0
    while True:
        # if RENDER: env.render()

        S = Variable(torch.from_numpy(s).type(torch.FloatTensor)).unsqueeze(0)
        V, A, (h_n, c_n) = agent.forward(S, (h_n, c_n))

        a_prob = A.data.numpy()[0]
        a = int(np.random.choice(a_prob.shape[0], p=a_prob))

        s_, r, done, info = env.step(a)
        s_ = blur(s_, BlurProb)
        ep_r += r

        # Redefine the reward function
        r = 0 if not done else -1

        s_lst.append(s)
        a_lst.append(a)
        r_lst.append(r)

        # Swap states
        s = s_

        # Update every T = 20 steps or episode ends (list len < 20)
        if len(s_lst) == 20 or done:

            # 1. Compute v(s_T)
            S_ = Variable(torch.from_numpy(s_).type(torch.FloatTensor)).unsqueeze(0)
            V_ = agent.forward(S_, (h_n, c_n))[0].data.numpy()[0][0]

            # Compute ^R_t (estimator for Q(s_t, a_t)
            n_step = []
            ret = V_ if not done else .0
            for r in reversed(r_lst):
                ret = r + .9 * ret
                n_step.append(ret)

            n_step = np.array(list(reversed(n_step)))[:, np.newaxis]

            S = Variable(torch.from_numpy(np.array(s_lst)).type(torch.FloatTensor))

            # Use blank hidden states
            V, A, _ = agent.forward(S, None)

            Adv = Variable(torch.from_numpy(n_step - V.data.numpy()).type(torch.FloatTensor))
            N_step = Variable(torch.from_numpy(np.array(n_step)).type(torch.FloatTensor))

            act = np.array(a_lst)[:, np.newaxis]
            Act = Variable(torch.from_numpy(act).type(torch.LongTensor))

            loss = -Adv * torch.log(A).gather(1, Act) + float(c) * (V - N_step) ** 2
            total_loss = loss.mean()   # optimize total loss

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            # Clear trackers
            s_lst, a_lst, r_lst = [], [], []

        if done:
            Ep_r.append(ep_r)
            print(ep_r)
            break

import pickle
with open('r_a2c_blur_cp_{0}.pkl'.format(args.file_index), 'wb') as handler:
    pickle.dump(Ep_r, handler, protocol=pickle.HIGHEST_PROTOCOL)

# plt.title('Learning Curve')
# plt.plot(Ep_r, label='c = {0}, lr = {1}'.format(c, lr))
# plt.legend(loc='best')
# plt.show()
