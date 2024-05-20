import gym
import numpy as np
import pickle
import click
from collections import deque

# Load the trained q-table (if needed for comparison)
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

# Reward function with tuning parameters
def get_reward(reward, goal_reward=20, step_penalty=-1, obstacle_penalty=-10):
    if reward == 20:  # assuming 20 is the reward for reaching the goal
        return goal_reward
    elif reward == -10:  # assuming -10 is the penalty for hitting an obstacle
        return obstacle_penalty
    else:
        return step_penalty  # penalty for each step taken

def bfs_search(env, goal_reward=20, step_penalty=-1, obstacle_penalty=-10):
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
            tuned_reward = get_reward(reward, goal_reward, step_penalty, obstacle_penalty)
            new_penalties = total_penalties + (1 if reward == -10 else 0)
            if done:
                return path + [action], total_reward + tuned_reward, new_penalties
            if next_state not in visited:
                queue.append((next_state, path + [action], total_reward + tuned_reward, new_penalties))
    return path, total_reward, total_penalties

def evaluate_algorithm(env, algorithm_fn, goal_reward, step_penalty, obstacle_penalty, episodes):
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0

    for _ in range(episodes):
        path, rewards, penalties = algorithm_fn(env, goal_reward, step_penalty, obstacle_penalty)
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
@click.option('--episodes', default=100, help='Number of episodes to run')
def evaluate_agent(env_name, episodes):
    env = gym.make(env_name)
    
    reward_combinations = [
        {'goal_reward': 20, 'step_penalty': -1, 'obstacle_penalty': -10},
        {'goal_reward': 50, 'step_penalty': -2, 'obstacle_penalty': -10},
        {'goal_reward': 20, 'step_penalty': -1, 'obstacle_penalty': -20}
    ]

    for i, rewards in enumerate(reward_combinations, start=1):
        print(f"Evaluating BFS with Reward Combination {i}: {rewards}")
        avg_timesteps, avg_rewards, avg_penalties = evaluate_algorithm(
            env, bfs_search, rewards['goal_reward'], rewards['step_penalty'], rewards['obstacle_penalty'], episodes
        )
        print(f"BFS Average Timesteps taken: {avg_timesteps}")
        print(f"BFS Average Rewards: {avg_rewards}")
        print(f"BFS Average Penalties: {avg_penalties}")
        print("\n")

if __name__ == '__main__':
    evaluate_agent()
