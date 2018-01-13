import matplotlib.pyplot as plt
import gym


Env = 'Enduro-ram-v0'
env = gym.make(Env)

# Print lives in Atari (Atari specific)
print('Lives', env.unwrapped.ale.lives(), '.\n')

# Print action meanings
action_lst = env.unwrapped.get_action_meanings()
print('Num. actions:', len(action_lst))
for action in action_lst: print(action)


for _ in range(1):
    env.reset()

    ep_steps = 0
    while True:

        # Print screen for every __ steps
        screen = env.render(mode='rgb_array')
        if ep_steps % 5 == 0:
            plt.imshow(screen)
            plt.show()

        # Print action taken
        a = env.action_space.sample()
        print(a)

        _, r, done, _ = env.step(a)

        # Print reward function
        if r != 0: print(r)

        ep_steps += 1
        if done:
            # Print total steps in an episode
            print(ep_steps)
            break
