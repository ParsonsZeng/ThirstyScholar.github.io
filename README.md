# Thirsty Scholar's Deep Thinking



This repo contains the following materials:



## Deep Deterministic Policy Gradient (DDPG)

A simple implementation of the *deep deterministic policy gradient* algorithm presented in [this paper](https://arxiv.org/pdf/1509.02971.pdf) on the classical [Pendulum task](https://github.com/openai/gym/wiki/Pendulum-v0) via [OpenAI Gym](https://gym.openai.com).

![ddpg_pendulum_result](DDPG/ddpg_pendulum_result.png)

I also normalized the state input by keeping a running stat using code from [John Schulman's repo](https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py) and scale the reward signal into the range roughly between [-1, 0] by dividing it by 16.

