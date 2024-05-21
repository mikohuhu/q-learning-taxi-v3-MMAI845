import gym
import numpy as np
import pickle

# Load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

def evaluate_algorithm(env, algorithm_fn, episodes):
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0

    for _ in range(episodes):
        path, rewards, penalties = algorithm_fn(env)
        total_timesteps += len(path)
        total_rewards += rewards
        total_penalties += penalties
        env.reset()

    average_timesteps = total_timesteps / episodes
    average_rewards = total_rewards / episodes
    average_penalties = total_penalties / episodes
    return average_timesteps, average_rewards, average_penalties

def evaluate_q_learning(env, q_table, episodes):
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, rewards = 0, 0, 0

        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            rewards += reward
            epochs += 1

        total_rewards += rewards
        total_timesteps += epochs
        total_penalties += penalties

    average_rewards = total_rewards / episodes
    average_timesteps = total_timesteps / episodes
    average_penalties = total_penalties / episodes
    return average_rewards, average_timesteps, average_penalties
