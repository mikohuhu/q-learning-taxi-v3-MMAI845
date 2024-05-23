import gym
import numpy as np
import pickle
from custom_taxi_env import CustomTaxiEnv  # Import the custom environment

def train_q_learning(env, alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes, render=False):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    all_rewards = []
    all_penalties = []
    all_timesteps = []

    for episode in range(episodes):
        state = env.reset()
        rewards, penalties, timesteps = 0, 0, 0
        done = False

        while not done:
            if render:
                env.render()

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, _ = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            rewards += reward
            state = next_state
            timesteps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        all_rewards.append(rewards)
        all_penalties.append(penalties)
        all_timesteps.append(timesteps)

        if render:
            env.render()
            print(f"Episode: {episode + 1}, Reward: {rewards}, Penalties: {penalties}, Timesteps: {timesteps}")

    return q_table, all_rewards, all_penalties, all_timesteps

if __name__ == "__main__":
    env = CustomTaxiEnv()  # Use the custom environment

    hyperparams = [
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.5, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.9, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.5, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000}
    ]

    for i, params in enumerate(hyperparams):
        print(f"Training with hyperparameters set {i + 1}")
        q_table, rewards, penalties, timesteps = train_q_learning(env, **params)

        # Evaluate the trained Q-learning agent
        average_reward = np.mean(rewards)
        average_timesteps = np.mean(timesteps)
        average_penalties = np.mean(penalties)

        print(f"Q-learning Average Rewards: {average_reward}")
        print(f"Q-learning Average Timesteps taken: {average_timesteps}")
        print(f"Q-learning Average Penalties incurred: {average_penalties}")

        # Save Q-table
        filename = f"q_table_{i + 1}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)

# /Users/mikohu/PycharmProjects/Queens_MMAI/MMAI845_individual/MMAI845_team/q-learning-taxi-v3-MMAI845/.venv/bin/python /Users/mikohu/PycharmProjects/Queens_MMAI/MMAI845_individual/MMAI845_team/q-learning-taxi-v3-MMAI845/qlearning_env1.py
# Training with hyperparameters set 1
# Q-learning Average Rewards: 12.542
# Q-learning Average Timesteps taken: 8.458
# Q-learning Average Penalties incurred: 0.0
# Training with hyperparameters set 2
# Q-learning Average Rewards: 14.335
# Q-learning Average Timesteps taken: 6.665
# Q-learning Average Penalties incurred: 0.0
# Training with hyperparameters set 3
# Q-learning Average Rewards: 12.483
# Q-learning Average Timesteps taken: 8.517
# Q-learning Average Penalties incurred: 0.0
# Training with hyperparameters set 4
# Q-learning Average Rewards: 13.8555
# Q-learning Average Timesteps taken: 7.1445
# Q-learning Average Penalties incurred: 0.0
# Training with hyperparameters set 5
# Q-learning Average Rewards: 13.9965
# Q-learning Average Timesteps taken: 7.0035
# Q-learning Average Penalties incurred: 0.0