import gym
import heapq


def a_star_search(env, state, render=False):
    priority_queue = [(0, state, [], 0, 0)]  # (cost, state, path, total_reward, total_penalties)
    visited = set()

    while priority_queue:
        cost, state, path, total_reward, total_penalties = heapq.heappop(priority_queue)
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
                heapq.heappush(priority_queue, (cost + reward, next_state, path + [action], total_reward + reward, new_penalties))

    return action, state, total_reward, total_penalties
