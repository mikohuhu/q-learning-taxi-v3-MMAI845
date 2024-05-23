import gym
import numpy as np
import pickle
from custom_taxi_env import CustomTaxiEnv  # Import the custom environment

def evaluate_agent(q_table, env, num_trials):
    total_epochs, total_penalties, total_rewards = 0, 0, 0

    for _ in range(num_trials):
        state = env.reset()
        epochs, num_penalties, rewards = 0, 0, 0

        while True:
            action = np.argmax(q_table[state])
            state, reward, done, _ = env.step(action)

            if reward == -10:
                num_penalties += 1

            rewards += reward
            epochs += 1

            if done:
                break

        total_penalties += num_penalties
        total_epochs += epochs
        total_rewards += rewards

    average_time = total_epochs / float(num_trials)
    average_penalties = total_penalties / float(num_trials)
    average_rewards = total_rewards / float(num_trials)

    return average_time, average_rewards, average_penalties


if __name__ == "__main__":
    env = CustomTaxiEnv()  # Use the custom environment
    num_trials = 10  # Number of evaluation trials per Q-table

    hyperparams = [
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.5, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.9, 'epsilon_min': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.5, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000}
    ]

    best_performance = float('-inf')
    best_hyperparams = None
    best_results = None

    for i, params in enumerate(hyperparams):
        print(f"Evaluating Q-table {i + 1}...")
        filename = f"q_table_{i + 1}.pickle"
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)

        average_time, average_rewards, average_penalties = evaluate_agent(q_table, env, num_trials)

        print(f"Q-table {i + 1} Evaluation Results:")
        print(f"Average Time Steps: {average_time}")
        print(f"Average Rewards: {average_rewards}")
        print(f"Average Penalties: {average_penalties}")
        print()

        # Update best hyperparameters if needed
        performance = average_rewards - average_penalties  # Consider both rewards and penalties
        if performance > best_performance:
            best_performance = performance
            best_hyperparams = params
            best_results = (average_time, average_rewards, average_penalties)

    print("Best Hyperparameters:")
    print(best_hyperparams)
    print("Best Results:")
    print(f"Average Time Steps: {best_results[0]}")
    print(f"Average Rewards: {best_results[1]}")
    print(f"Average Penalties: {best_results[2]}")

# /Users/mikohu/PycharmProjects/Queens_MMAI/MMAI845_individual/MMAI845_team/q-learning-taxi-v3-MMAI845/.venv/bin/python /Users/mikohu/PycharmProjects/Queens_MMAI/MMAI845_individual/MMAI845_team/q-learning-taxi-v3-MMAI845/qlearning_hyper_evaluate_env1.py
# Evaluating Q-table 1...
# Q-table 1 Evaluation Results:
# Average Time Steps: 4.0
# Average Rewards: 17.0
# Average Penalties: 0.0
#
# Evaluating Q-table 2...
# Q-table 2 Evaluation Results:
# Average Time Steps: 3.1
# Average Rewards: 17.9
# Average Penalties: 0.0
#
# Evaluating Q-table 3...
# Q-table 3 Evaluation Results:
# Average Time Steps: 3.8
# Average Rewards: 17.2
# Average Penalties: 0.0
#
# Evaluating Q-table 4...
# Q-table 4 Evaluation Results:
# Average Time Steps: 3.9
# Average Rewards: 17.1
# Average Penalties: 0.0
#
# Evaluating Q-table 5...
# Q-table 5 Evaluation Results:
# Average Time Steps: 3.9
# Average Rewards: 17.1
# Average Penalties: 0.0
#
# Best Hyperparameters:
# {'alpha': 0.5, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.1, 'episodes': 2000}
# Best Results:
# Average Time Steps: 3.1
# Average Rewards: 17.9
# Average Penalties: 0.0
#
# Process finished with exit code 0
