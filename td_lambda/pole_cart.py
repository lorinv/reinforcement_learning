import gym
from gym import envs

env = gym.make('CartPole-v1')
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while 1:
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
		    print("Episode finished after {} timesteps".format(t+1))
		    break