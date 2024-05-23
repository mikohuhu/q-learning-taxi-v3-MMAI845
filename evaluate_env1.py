import gym
import numpy as np
import pickle
from custom_taxi_env import CustomTaxiEnv  # Import the custom environment

def evaluate_agent(q_table, env, num_episodes):
    total_epochs, total_penalties, total_rewards = 0, 0, 0

    for _ in range(num_episodes):
        state = env.reset()
        epochs, num_penalties, episode_reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _ = env.step(action)

            if reward == -10:
                num_penalties += 1

            episode_reward += reward
            epochs += 1

        total_penalties += num_penalties
        total_epochs += epochs
        total_rewards += episode_reward

    average_time = total_epochs / num_episodes
    average_penalties = total_penalties / num_episodes
    average_rewards = total_rewards / num_episodes

    return average_time, average_rewards, average_penalties

if __name__ == "__main__":
    env = CustomTaxiEnv()  # Use the custom environment
    episodes = 10  # Number of episodes to evaluate

    # Load the Q-table
    with open('q_table_env1_step1.pickle', 'rb') as f:
        q_table = pickle.load(f)

    avg_time_steps, avg_rewards, avg_penalties = evaluate_agent(q_table, env, episodes)

    print(f"Average time steps taken: {avg_time_steps}")
    print(f"Average rewards obtained: {avg_rewards}")
    print(f"Average number of penalties incurred: {avg_penalties}")

#Results from the following training hyperparameters:
#episodes, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1
# Average time steps taken: 3.2
# Average rewards obtained: 17.8
# Average number of penalties incurred: 0.0
#
# Process finished with exit code 0

