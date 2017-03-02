import random
import numpy as np
import sys
sys.path.append("../gym/")
import gym
from time import sleep
from collections import defaultdict

MAX_STEPS = 200
NUM_EPISODES = 200
NUM_PLAYS = 10

DISCOUNT = .99

'''
#...Fix the first-visit -- update the state,action pair every time that pair occures -- use
    the value of the first occurance for every update
    Instead of just adding all the rewards -- dicount all the rewards DISCOUNT^DISTANCE AWAY
'''

class MC_Q:
    
    def __init__(self, num_states, num_actions):
        self.state_action_values = np.random.rand(num_states,num_actions)                   

    def get_next_action(self, state, e=50):
        val = random.randint(0,100)
        if val < e:
            return random.randint(0,len(self.state_action_values[state]) - 1)
        else:
            return np.argmax(self.state_action_values[state])    

    def operation(self, env):         
        returns_sum = defaultdict(float)
        num_returns = defaultdict(float)   
        average_reward = 0 
        reward_list = []

        for episode in range(NUM_EPISODES):
            steps = 0
            returns = []
            reward_sequence = []            
            observation = env.reset()
            while steps < MAX_STEPS:
                action = self.get_next_action(observation)
                returns.append((observation,action))
                observation, reward, done, info = env.step(action)
                reward_sequence.append(reward)
                if done:
                    break       
            
            #Find the first occurance for each state-action pair for this
            #episode
            discounted_reward = {}
            for i in range(len(returns)):
                if returns[i] not in discounted_reward.keys():
                    r = 0
                    for j, val in enumerate(reward_sequence[i:]):
                        r += val * (DISCOUNT**(j+1))
                    discounted_reward[returns[i]] = r
            
            #For every occurance of a state, add the discounted value of the first state
            #Keep count of how many times you have seen each state
            for i in range(len(returns)):
                state,action = returns[i]
                returns_sum[returns[i]] += discounted_reward[returns[i]]
                num_returns[returns[i]] += 1                               
                self.state_action_values[state][action] = returns_sum[returns[i]] / num_returns[returns[i]]
        
            print "Episode %d" % episode
            final_timesteps, final_reward = self.play_game(env)
            #average_reward += final_reward
            reward_list.append(final_reward)
        print
        print

        print "Reward List: %s" % str(reward_list) 
        for s in range(12):
            print "State: %d" % s
            for a in range(4):                
                print "\tAction: %d \t Value: %lf" % (a, self.state_action_values[s][a])
            print

        #print "Average Reward: %s" % str(average_reward / NUM_EPISODES) 
        #print "Num Episodes: %d" % NUM_EPISODES

    def play_game(self, env):
        final_reward = 0
        final_timesteps = 0
        observation = env.reset()        
        for t in range(MAX_STEPS):            
            #sleep(1)
            #env.render()
            action = self.get_next_action(observation, e=0)
            observation, reward, done, info = env.step(action)
            final_reward += reward
            final_timesteps += 1
            if done:                
                break

        print "Total Timesteps: %d" % final_timesteps
        print "Final Observation: %d" % observation
        print "Final Reward: %d" % final_reward
        #raw_input("")
        return final_timesteps, final_reward

if __name__ == '__main__':
    from envs.mdp_gridworld import MDPGridworldEnv
    env_names = ['Grid World', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'Taxi-v2']
    env_list = [MDPGridworldEnv(),gym.make('FrozenLake-v0').env, gym.make('FrozenLake8x8-v0').env, gym.make('Taxi-v2').env]
    env = env_list[0]
    random.seed()
    #for j, env in enumerate(env_list):        
    #    total_ts = []
    #    total_reward = []
    #    for i in range(NUM_PLAYS):
    mc = MC_Q(env.observation_space.n, env.action_space.n)
    mc.operation(env)
            #ts, reward = mc.play_game(env)
            #total_reward.append(reward)
            #total_ts.append(ts)
    '''
    print 
    print "************************************************"
    print "Game: %s" % env_names[j]
    print "Average Reward: %s" % str(np.average(total_reward)) 
    print "Max Reward: %s" % str(np.max(total_reward))
    print "Average Timesteps: %s" % str(np.average(total_ts))
    print "**************************************************"
    print

    raw_input("")
    '''        
    

