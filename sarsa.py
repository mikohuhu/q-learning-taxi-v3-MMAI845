import gym
import numpy as np
import pickle

import gym
import numpy as np

def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)

def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, render=False, seed=None):
    if seed is not None:
        set_seed(env, seed)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    all_rewards = []
    all_penalties = []
    all_timesteps = []

    for episode in range(episodes):
        state = env.reset()
        rewards, penalties, timesteps = 0, 0, 0
        done = False

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        while not done:
            if render:
                env.render()

            next_state, reward, done, _ = env.step(action)

            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()  # Explore action space
            else:
                next_action = np.argmax(q_table[next_state])  # Exploit learned values

            old_value = q_table[state, action]
            next_value = q_table[next_state, next_action]
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            rewards += reward
            state = next_state
            action = next_action
            timesteps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        all_rewards.append(rewards)
        all_penalties.append(penalties)
        all_timesteps.append(timesteps)

    return q_table, all_rewards, all_penalties, all_timesteps