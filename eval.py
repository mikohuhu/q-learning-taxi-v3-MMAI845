import gym
import numpy as np
import pickle
import numpy as np


# Load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

def evaluate_algorithm(env, algorithm, episodes, render=False):
    total_timesteps = 0
    total_rewards = 0
    total_penalties = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        timesteps = 0
        rewards = 0
        penalties = 0
        
        while not done:
            if render:
                env.render()
            
            action, next_state, reward, penalty = algorithm(env, state, render=render)  # Pass render parameter
            state = next_state

            timesteps += 1
            rewards += reward
            penalties += penalty
            if reward != 0:  # Typically, rewards are given at the end of an episode
                done = True
        
        total_timesteps += timesteps
        total_rewards += rewards
        total_penalties += penalties
    
    average_timesteps = total_timesteps / episodes
    average_rewards = total_rewards / episodes
    average_penalties = total_penalties / episodes
    
    return average_timesteps, average_rewards, average_penalties


def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
def evaluate_sarsa(env, q_table, episodes, render=False, seed=None):
    if seed is not None:
        set_seed(env, seed)
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        timesteps = 0
        rewards = 0
        penalties = 0

        action = np.argmax(q_table[state])  # Always exploit the learned policy
        
        while not done:
            if render:
                env.render()

            next_state, reward, done, info = env.step(action)
            next_action = np.argmax(q_table[next_state])  # Always exploit the learned policy

            state = next_state
            action = next_action

            timesteps += 1
            rewards += reward
            if reward == -10:
                penalties += 1

        total_rewards += rewards
        total_timesteps += timesteps
        total_penalties += penalties
    
    average_rewards = total_rewards / episodes
    average_timesteps = total_timesteps / episodes
    average_penalties = total_penalties / episodes
    
    return average_rewards, average_timesteps, average_penalties


def evaluate_q_learning(env, q_table, episodes, render=False):
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        timesteps = 0
        rewards = 0
        penalties = 0
        
        while not done:
            if render:
                env.render()
            
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            
            timesteps += 1
            rewards += reward
            if reward == -10:
                penalties += 1
        
        total_rewards += rewards
        total_timesteps += timesteps
        total_penalties += penalties
    
    average_rewards = total_rewards / episodes
    average_timesteps = total_timesteps / episodes
    average_penalties = total_penalties / episodes
    
    return average_rewards, average_timesteps, average_penalties