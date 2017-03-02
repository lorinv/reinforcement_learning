'''
Lorin Vandegrift
11354621
02/26/2017
'''
import sys
from time import sleep
import numpy as np
sys.path.append("../gym/")
import gym
import random

DISCOUNT = .7
TRANSISTION_PROB = 0
NEXT_STATE = 1
REWARD = 2
TERMINAL = 3


def initialize_policy(env):
    num_states = env.observation_space.n
    policy = [1] * num_states
    state_values = [.5] * num_states

    return policy, state_values 

def train_mdp(env, policy, state_values):
    episodes = 20
    num_states = env.observation_space.n
    num_actions = env.action_space.n            
    
    for num_updates in range(episodes): 
        steps = 0
        observation = env.reset()
        visited_states = [observation]
        reward_sequence = [0]   
        while steps < 50:
            action = find_next_action(env, state_values, observation)
            observation, reward, done, info = env.step(action)
            print "Episode: %d" % num_updates
            print "Step: %d" % steps
            print "Observation: %s" % str(observation)
            print "Reward: %s" % str(reward)
            print "Terminal: %s" % str(done)
            print "State Values: %s" % str(state_values)
            print
            print
            visited_states.append(observation)
            reward_sequence.append(reward)
            steps += 1
            sleep(.1)
            if done == True:
                break

        state_values = evaluate_state_values(state_values, num_updates + 1, visited_states, reward_sequence)


    return state_values

def evaluate_state_values(state_values, num_updates, visited_states, reward_sequence):
    dstates = []    
    for i in range(len(visited_states)):
        if visited_states[i] not in dstates: 
            dstates.append(visited_states[i])
            r = sum(reward_sequence[i:])
            state_values[visited_states[i]] += (r - state_values[visited_states[i]]) / num_updates
    
    return state_values


def find_next_action(env, values, current_state):
    s = 0
    if random.randint(0, 100) < 10:
        return env.action_space.sample()
    else:
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
        sleep(1)
        env.render()
        print(observation)
        action = find_next_action(env, values, observation)
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
    random.seed()
    env_names = ['Grid World', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'Taxi-v2']
    env_list = [MDPGridworldEnv(),gym.make('FrozenLake-v0').env, gym.make('FrozenLake8x8-v0').env, gym.make('Taxi-v2').env]
    result_list = []
    policy, state_values = initialize_policy(env_list[0])
    state_values = train_mdp(env_list[0], policy, state_values)
    play_game(state_values, env_list[0])
    '''
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
    '''
