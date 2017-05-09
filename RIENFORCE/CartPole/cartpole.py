import gym
from agent import PolicyGradient

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, vanilla Policy gradient has high variance
env = env.unwrapped

# # Basic info about the envir
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
)


running_reward = 0
for i_episode in range(3000):
    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        RL.store_reward(reward)

        ep_rs_sum = sum(RL.ep_rs)
        if done or ep_rs_sum > 2 * DISPLAY_REWARD_THRESHOLD:  # End the ep if done or get enough reward
            running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True

            print("Episode:", i_episode, "  Reward:", int(running_reward))
            vt = RL.learn()

            break

        observation = observation_

        if running_reward > 2 * DISPLAY_REWARD_THRESHOLD: break
