import os.path
import sys
import gym
from gym import wrappers
from agent import PGPE


mod_path = os.path.dirname(os.path.abspath(sys.argv[0]))
save_path = os.path.join(mod_path, 'cartpole_experiment_1')

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, save_path, force=True)  # Gym's built-in monitor functionality

agent = PGPE(
    n_features=env.observation_space.shape[0],
    n_actions=env.action_space.n
)


for ep in range(200):
    observation = env.reset()

    while True:
        # A typical RL env-agent paradigm
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        agent.store_reward(reward)

        if done:
            print("Episode:", ep, "  Reward:", int(agent.get_return()))
            vt = agent.learn_and_sample()  # learn after an ep ends
            break
        
        # Swap obs
        observation = observation_

# Close env
# If not, can't use gym's built-in monitor funcationality
env.close()
