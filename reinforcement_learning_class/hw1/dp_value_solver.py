'''
Lorin Vandegrift
11354621
02/18/2017
'''

import sys
from time import sleep
import numpy as np
sys.path.append("../gym/")
import gym

TRANSISTION_PROB = 0
NEXT_STATE = 1
REWARD = 2
TERMINAL = 3

DISCOUNT = .99

def train_mdp(env):
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    state_values = [0] * NUM_STATES
    THESHOLD = False

    count = 0
    while THESHOLD == False:
        count += 1
        for curr_state in range(NUM_STATES):
            state_deltas = [0] * NUM_STATES
            old_state_value = state_values[curr_state]
            max_action_value = 0            
            
            for curr_action in range(NUM_ACTIONS):
                curr_action_value = 0

                for p_next_state in env.P[curr_state][curr_action]:
                    transition_prob = p_next_state[0]
                    resulting_state = p_next_state[1]
                    reward = p_next_state[2]                
                    curr_action_value += transition_prob \
                        * (reward + DISCOUNT*state_values[resulting_state])      

                if curr_action_value > max_action_value:
                    max_action_value = curr_action_value

            state_values[curr_state] = max_action_value

        if count > 1000:
            THESHOLD = True

    return state_values


def find_next_action(values, current_state):
    max_value = 0
    max_action = 0
    for action in env.P[current_state]:
        v = 0
        for possible_state in env.P[current_state][action]:
            v += possible_state[0] * (possible_state[REWARD] + DISCOUNT*values[possible_state[NEXT_STATE]])
        if v > max_value:
            max_value = v
            max_action = action

    return max_action

def play_game(values, env):
        final_reward = 0
        final_timesteps = 0
        observation = env.reset()
        for t in range(100):
            #sleep(1)
            env.render()
            print(observation)                        
            action = find_next_action(values, observation)
            observation, reward, done, info = env.step(action)
            final_reward += reward
            if done:
                print
                print("Episode finished after {} timesteps".format(t+1))
                final_timesteps = t+1
                print("Final Reward Value: %d" % final_reward)
                print
                break

        return final_timesteps, final_reward 

def print_results(result_list):
    print "FINAL RESULTS:"
    for i, result in enumerate(result_list):        
        print
        print "***************************************"
        print "World: %s" % env_names[i]
        print "Average Time Steps: %s" % str(result_list[i][0])
        print "Average Reward: %s" % str(result_list[i][1])  
        print "Max Reward: %s" % str(result_list[i][2])  
        print "Number of Attempts: %s" % str(result_list[i][3])
        print "****************************************"


if __name__ == '__main__':
    # line above ensures the interpreter can locate envs.mdp_gridworld
    from envs.mdp_gridworld import MDPGridworldEnv    
    env_names = ['Grid World', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'Taxi-v2']
    env_list = [MDPGridworldEnv(),gym.make('FrozenLake-v0').env, gym.make('FrozenLake8x8-v0').env, gym.make('Taxi-v2').env]
    result_list = []
    for env in env_list:        
        final_reward = 0
        final_timesteps = 0
        count = 0
        NUM_ITER = 50
        max_reward = 0
        for i_episode in range(NUM_ITER):                        
            values = train_mdp(env)        
            timesteps, reward = play_game(values, env)
            if reward > max_reward:
                max_reward = reward
            final_reward += reward
            final_timesteps += timesteps
            count += 1
        avg_reward = float(final_reward) / float(count)
        avg_timesteps = float(final_timesteps) / float(count)
        result_list.append((avg_timesteps, avg_reward, max_reward, NUM_ITER))

    print_results(result_list)
        








