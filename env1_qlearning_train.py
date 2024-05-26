#Customized Environment Qlearning Training, default parameters

import gym
import numpy as np
import pickle
from custom_taxi_env import CustomTaxiEnv  # Import the custom environment

def train_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, render=False):
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

    with open('q_table_env1_step1.pickle', 'wb') as f:
        pickle.dump(q_table, f)

    return q_table, all_rewards, all_penalties, all_timesteps

if __name__ == "__main__":
    env = CustomTaxiEnv()  # Use the custom environment
    episodes = 10000  # Increase the number of episodes for better training
    render_training = False  # Set this to True to enable rendering

    q_table, rewards, penalties, timesteps = train_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, render=render_training)

    # Evaluate the trained Q-learning agent
    average_reward = np.mean(rewards)
    average_timesteps = np.mean(timesteps)
    average_penalties = np.mean(penalties)

    print(f"Q-learning Average Rewards: {average_reward}")
    print(f"Q-learning Average Timesteps taken: {average_timesteps}")
    print(f"Q-learning Average Penalties incurred: {average_penalties}")


# Q-learning Average Rewards: 15.8738
# Q-learning Average Timesteps taken: 5.1262
# Q-learning Average Penalties incurred: 0.0