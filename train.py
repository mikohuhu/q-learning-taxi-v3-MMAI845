import gym
import numpy as np
import pickle
import click
from collections import deque
import heapq

# load the trained q-table
with open('q_table.pickle', 'rb') as f:
    q_table = pickle.load(f)

def bfs_search(env):
    initial_state = env.reset()
    queue = deque([(initial_state, [])])
    visited = set()

    while queue:
        state, path = queue.popleft()
        if state in visited:
            continue
        visited.add(state)

        for action in range(env.action_space.n):
            env.env.s = state  # Set the environment to the current state
            next_state, reward, done, _ = env.step(action)
            if done:
                return path + [action]
            if next_state not in visited:
                queue.append((next_state, path + [action]))

def a_star_search(env):
    initial_state = env.reset()
    priority_queue = [(0, initial_state, [])]
    visited = set()

    while priority_queue:
        cost, state, path = heapq.heappop(priority_queue)
        if state in visited:
            continue
        visited.add(state)

        for action in range(env.action_space.n):
            env.env.s = state  # Set the environment to the current state
            next_state, reward, done, _ = env.step(action)
            if done:
                return path + [action]
            if next_state not in visited:
                heapq.heappush(priority_queue, (cost + reward, next_state, path + [action]))

@click.command()
@click.option('--env_name', default="Taxi-v3", help='Gym environment name')
def evaluate_agent(env_name):
    env = gym.make(env_name)
    
    print("Evaluating BFS:")
    bfs_path = bfs_search(env)
    bfs_timesteps = len(bfs_path)
    print("BFS Timesteps taken: {}".format(bfs_timesteps))

    env.reset()  # Reset the environment to initial state for the next algorithm

    print("Evaluating A*:")
    a_star_path = a_star_search(env)
    a_star_timesteps = len(a_star_path)
    print("A* Timesteps taken: {}".format(a_star_timesteps))

    # Existing Q-learning evaluation
    state = env.reset()
    epochs, penalties, rewards = 0, 0, 0
    frames = []  # for animation

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })

        epochs += 1

    print("Q-learning Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))

if __name__ == '__main__':
    evaluate_agent()
