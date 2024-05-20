import gym
import numpy as np
import pickle
import click
from collections import deque
import heapq

# Load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

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
            new_penalties = total_penalties + (1 if reward == -10 else 0)
            # print(f"BFS Step: state={state}, action={action}, next_state={next_state}, reward={reward}, penalties={new_penalties}")
            if done:
                # print(f"BFS Path: {path + [action]}, Total Reward: {total_reward + reward}, Total Penalties: {new_penalties}")
                return path + [action], total_reward + reward, new_penalties
            if next_state not in visited:
                queue.append((next_state, path + [action], total_reward + reward, new_penalties))
    return path, total_reward, total_penalties

def a_star_search(env):
    initial_state = env.reset()
    priority_queue = [(0, initial_state, [], 0, 0)]  # (cost, state, path, total_reward, total_penalties)
    visited = set()

    while priority_queue:
        cost, state, path, total_reward, total_penalties = heapq.heappop(priority_queue)
        if state in visited:
            continue
        visited.add(state)

        for action in range(env.action_space.n):
            env.env.s = state  # Set the environment to the current state
            next_state, reward, done, _ = env.step(action)
            new_penalties = total_penalties + (1 if reward == -10 else 0)
            # print(f"A* Step: state={state}, action={action}, next_state={next_state}, reward={reward}, penalties={new_penalties}")
            if done:
                # print(f"A* Path: {path + [action]}, Total Reward: {total_reward + reward}, Total Penalties: {new_penalties}")
                return path + [action], total_reward + reward, new_penalties
            if next_state not in visited:
                heapq.heappush(priority_queue, (cost + reward, next_state, path + [action], total_reward + reward, new_penalties))
    return path, total_reward, total_penalties

@click.command()
@click.option('--env_name', default="Taxi-v3", help='Gym environment name')
@click.option('--episodes', default=100, help='Number of episodes to run')
def evaluate_agent(env_name, episodes):
    env = gym.make(env_name)

    def evaluate_algorithm(algorithm_fn):
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

    print("Evaluating BFS:")
    bfs_average_timesteps, bfs_average_rewards, bfs_average_penalties = evaluate_algorithm(bfs_search)
    print("BFS Average Timesteps taken: {}".format(bfs_average_timesteps))
    print("BFS Average Rewards: {}".format(bfs_average_rewards))
    print("BFS Average Penalties: {}".format(bfs_average_penalties))

    print("Evaluating A*:")
    a_star_average_timesteps, a_star_average_rewards, a_star_average_penalties = evaluate_algorithm(a_star_search)
    print("A* Average Timesteps taken: {}".format(a_star_average_timesteps))
    print("A* Average Rewards: {}".format(a_star_average_rewards))
    print("A* Average Penalties: {}".format(a_star_average_penalties))

    def evaluate_q_learning():
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

    print("Evaluating Q-learning:")
    q_learning_avg_rewards, q_learning_avg_timesteps, q_learning_avg_penalties = evaluate_q_learning()
    print("Q-learning Average Rewards: {}".format(q_learning_avg_rewards))
    print("Q-learning Average Timesteps taken: {}".format(q_learning_avg_timesteps))
    print("Q-learning Average Penalties incurred: {}".format(q_learning_avg_penalties))

if __name__ == '__main__':
    evaluate_agent()
