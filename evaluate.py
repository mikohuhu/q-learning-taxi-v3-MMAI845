import click
import gym
import numpy as np
import pickle
from BFS_search import bfs_search  # Import the modified BFS search function
from Astar_search import a_star_search  # Import the modified A* search function
from eval import evaluate_algorithm, evaluate_q_learning, evaluate_sarsa
from sarsa import sarsa
# Load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

@click.command()
@click.option('--env_name', default="Taxi-v3", help='Gym environment name')
@click.option('--episodes', default=100, help='Number of episodes to run')
@click.option('--render', is_flag=True, help='Render the environment')
def evaluate_agent(env_name, episodes, render):
    env = gym.make(env_name)

    print("Evaluating BFS:")
    bfs_average_timesteps, bfs_average_rewards, bfs_average_penalties = evaluate_algorithm(env, bfs_search, episodes, render)
    print("BFS Average Timesteps taken: {}".format(bfs_average_timesteps))
    print("BFS Average Rewards: {}".format(bfs_average_rewards))
    print("BFS Average Penalties: {}".format(bfs_average_penalties))

    print("Evaluating A*:")
    a_star_average_timesteps, a_star_average_rewards, a_star_average_penalties = evaluate_algorithm(env, a_star_search, episodes, render)
    print("A* Average Timesteps taken: {}".format(a_star_average_timesteps))
    print("A* Average Rewards: {}".format(a_star_average_rewards))
    print("A* Average Penalties: {}".format(a_star_average_penalties))

    print("Evaluating Q-learning:")
    q_learning_avg_rewards, q_learning_avg_timesteps, q_learning_avg_penalties = evaluate_q_learning(env, q_table, episodes, render)
    print("Q-learning Average Timesteps taken: {}".format(q_learning_avg_timesteps))
    print("Q-learning Average Rewards: {}".format(q_learning_avg_rewards))
    print("Q-learning Average Penalties incurred: {}".format(q_learning_avg_penalties))

    # print("Evaluating SARSA:")
    # sarsa_avg_rewards, sarsa_avg_timesteps, sarsa_avg_penalties = evaluate_sarsa(env, q_table, episodes, render)
    # print("SARSA Average Timesteps taken: {}".format(sarsa_avg_timesteps))
    # print("SARSA Average Rewards: {}".format(sarsa_avg_rewards))
    # print("SARSA Average Penalties incurred: {}".format(sarsa_avg_penalties))

if __name__ == '__main__':
    evaluate_agent()
