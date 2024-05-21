import click
import gym
import numpy as np
import pickle
from BFS_search import bfs_search
from Astar_search import a_star_search
from eval import evaluate_algorithm, evaluate_q_learning

# Load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

@click.command()
@click.option('--env_name', default="Taxi-v3", help='Gym environment name')
@click.option('--episodes', default=100, help='Number of episodes to run')
def evaluate_agent(env_name, episodes):
    env = gym.make(env_name)

    print("Evaluating BFS:")
    bfs_average_timesteps, bfs_average_rewards, bfs_average_penalties = evaluate_algorithm(env, bfs_search, episodes)
    print("BFS Average Timesteps taken: {}".format(bfs_average_timesteps))
    print("BFS Average Rewards: {}".format(bfs_average_rewards))
    print("BFS Average Penalties: {}".format(bfs_average_penalties))

    print("Evaluating A*:")
    a_star_average_timesteps, a_star_average_rewards, a_star_average_penalties = evaluate_algorithm(env, a_star_search, episodes)
    print("A* Average Timesteps taken: {}".format(a_star_average_timesteps))
    print("A* Average Rewards: {}".format(a_star_average_rewards))
    print("A* Average Penalties: {}".format(a_star_average_penalties))

    print("Evaluating Q-learning:")
    q_learning_avg_rewards, q_learning_avg_timesteps, q_learning_avg_penalties = evaluate_q_learning(env, q_table, episodes)
    print("Q-learning Average Rewards: {}".format(q_learning_avg_rewards))
    print("Q-learning Average Timesteps taken: {}".format(q_learning_avg_timesteps))
    print("Q-learning Average Penalties incurred: {}".format(q_learning_avg_penalties))

if __name__ == '__main__':
    evaluate_agent()
