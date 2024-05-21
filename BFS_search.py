import gym
from collections import deque


def bfs_search(env, state, render=False):
    queue = deque([(state, [], 0, 0)])  # (state, path, total_reward, total_penalties)
    visited = set()

    while queue:
        state, path, total_reward, total_penalties = queue.popleft()
        if state in visited:
            continue
        visited.add(state)

        for action in range(env.action_space.n):
            env.env.s = state  # Set the environment to the current state
            if render:
                env.render()  # Render the environment

            next_state, reward, done, _ = env.step(action)
            new_penalties = total_penalties + (1 if reward == -10 else 0)
            if done:
                if render:
                    env.render()  # Render the environment on reaching the goal state
                return action, next_state, total_reward + reward, new_penalties
            if next_state not in visited:
                queue.append((next_state, path + [action], total_reward + reward, new_penalties))

    return action, state, total_reward, total_penalties