import gym
import os.path

env_id = 'CartPole-v0'

#
script_path = os.path.dirname(__file__)
save_path = os.path.join(script_path, env_id)
print('Save to:', save_path)

# Monitor env. wrapper
env = gym.make(env_id)
env = gym.wrappers.Monitor(env, save_path)

# Roll out sample policy for demo
for _ in range(10):
    env.reset()

    while True:
        a = env.action_space.sample()
        _, _, done, _ = env.step(a)

        if done: break
