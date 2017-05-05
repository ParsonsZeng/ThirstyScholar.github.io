import os.path
import sys
import gym
from gym import wrappers
from agent import PGPE


mod_path = os.path.dirname(os.path.abspath(sys.argv[0]))
save_path = os.path.join(mod_path, 'cartpole_experiment_1')

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, save_path, force=True)

RL = PGPE(
    n_features=env.observation_space.shape[0],
    n_actions=env.action_space.n
)


for ep in range(200):
    observation = env.reset()

    while True:
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        RL.store_return(reward)
        r = RL.get_reward()

        if done:
            vt = RL.learn_and_sample()
            print("Episode:", ep, "  Reward:", int(r))
            break

        observation = observation_

# Close env.
env.close()
