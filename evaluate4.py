import gym
import numpy as np
import pickle
import click
from collections import deque
import heapq

# Load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

reward_combination = {'goal_reward': 20, 'step_penalty': -1, 'obstacle_penalty': -20}

alpha_values = [0.1, 0.2, 0.5]
gamma_values = [0.6, 0.8, 0.9]
epsilon_values = [0.1, 0.2, 0.3]
episode_values = [1000, 10, 100]

def bfs_search(env):
    initial_state = env.reset()
    queue = deque([(initial_state, [], 0, 0)])  # (state, path, total_reward, total_penalties)
    visited = set()

    while queue:
        state, path, total_reward, total_penalties = queue.popleft()
        if state in visited:
            continue
        visited.add(state)

        for action in range(env.action_space.n):
            env.env.s = state  # Set the environment to the current state
            next_state, reward, done, _ = env.step(action)
            new_penalties = total_penalties + (1 if reward == reward_combination['obstacle_penalty'] else 0)
            if done:
                return path + [action], total_reward + reward, new_penalties
            if next_state not in visited:
                queue.append((next_state, path + [action], total_reward + reward, new_penalties))
    return path, total_reward, total_penalties

def evaluate_bfs(env, episodes):
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0

    for _ in range(episodes):
        path, rewards, penalties = bfs_search(env)
        total_timesteps += len(path)
        total_rewards += rewards
        total_penalties += penalties
        env.reset()

    average_timesteps = total_timesteps / episodes
    average_rewards = total_rewards / episodes
    average_penalties = total_penalties / episodes
    return average_timesteps, average_rewards, average_penalties

@click.command()
@click.option('--env_name', default="Taxi-v3", help='Gym environment name')
def grid_search(env_name):
    env = gym.make(env_name)

    best_hyperparameters = None
    best_average_timesteps = float('inf')
    best_average_rewards = float('-inf')
    best_average_penalties = float('inf')

    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                for episodes in episode_values:
                    print(f"Evaluating BFS with alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={episodes}")
                    average_timesteps, average_rewards, average_penalties = evaluate_bfs(env, episodes)
                    print(f"Average Timesteps taken: {average_timesteps}")
                    print(f"Average Rewards: {average_rewards}")
                    print(f"Average Penalties: {average_penalties}")

                    # Criteria for best hyperparameters (can be adjusted)
                    if average_timesteps < best_average_timesteps:
                        best_hyperparameters = (alpha, gamma, epsilon, episodes)
                        best_average_timesteps = average_timesteps
                        best_average_rewards = average_rewards
                        best_average_penalties = average_penalties

    print("Best Hyperparameters:")
    print(f"Alpha: {best_hyperparameters[0]}, Gamma: {best_hyperparameters[1]}, Epsilon: {best_hyperparameters[2]}, Episodes: {best_hyperparameters[3]}")
    print(f"Best Average Timesteps taken: {best_average_timesteps}")
    print(f"Best Average Rewards: {best_average_rewards}")
    print(f"Best Average Penalties: {best_average_penalties}")

if __name__ == '__main__':
    grid_search()
