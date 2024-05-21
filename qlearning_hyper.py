import gym
import numpy as np
import pickle
import argparse

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_combination):
        super(CustomRewardWrapper, self).__init__(env)
        self.reward_combination = reward_combination

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.reward_combination == 3:
            # Combination 3: Original rewards and penalties
            if reward == -10:  # illegal move
                reward = -10
            elif reward == 20:  # successful drop-off
                reward = 20
            elif reward == -1:  # every timestep
                reward = -1

        return state, reward, done, info

def q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, _ = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            # Q-learning formula
            q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table

def evaluate_q_learning(env, q_table, episodes, render=False):
    total_rewards = 0
    total_timesteps = 0
    total_penalties = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        timesteps = 0
        rewards = 0
        penalties = 0
        
        while not done:
            if render:
                env.render()
            
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            
            timesteps += 1
            rewards += reward
            if reward == -10:
                penalties += 1
        
        total_rewards += rewards
        total_timesteps += timesteps
        total_penalties += penalties
    
    average_rewards = total_rewards / episodes
    average_timesteps = total_timesteps / episodes
    average_penalties = total_penalties / episodes
    
    return average_rewards, average_timesteps, average_penalties

def evaluate_custom(env_name, q_table, episodes, reward_combination, render=False):
    env = CustomRewardWrapper(gym.make(env_name), reward_combination=reward_combination)
    avg_rewards, avg_timesteps, avg_penalties = evaluate_q_learning(env, q_table, episodes, render)
    env.close()
    return avg_rewards, avg_timesteps, avg_penalties

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-learning with different hyperparameters')
    parser.add_argument('--env_name', type=str, default="Taxi-v3", help='Gym environment name')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of evaluation episodes')
    args = parser.parse_args()

    env_name = args.env_name
    eval_episodes = args.eval_episodes

    # Hyperparameter sets
    hyperparams = [
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'min_epsilon': 0.1, 'episodes': 2000},
        {'alpha': 0.5, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'min_epsilon': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'min_epsilon': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.9, 'min_epsilon': 0.1, 'episodes': 2000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.5, 'epsilon_decay': 0.995, 'min_epsilon': 0.1, 'episodes': 2000}
    ]

    for idx, params in enumerate(hyperparams, start=1):
        print(f"Training with Hyperparameter Set {idx}: {params}")
        env = CustomRewardWrapper(gym.make(env_name), reward_combination=3)
        q_table = q_learning(
            env,
            episodes=params['episodes'],
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon=params['epsilon'],
            epsilon_decay=params['epsilon_decay'],
            min_epsilon=params['min_epsilon']
        )
        
        # Save the trained Q-table
        with open(f'q_table_comb3_set{idx}.pickle', 'wb') as f:
            pickle.dump(q_table, f)

        # Evaluate the trained Q-table
        print(f"Evaluating Custom Environment with Reward Combination 3 using Hyperparameter Set {idx}:")
        avg_rewards, avg_timesteps, avg_penalties = evaluate_custom(env_name, q_table, eval_episodes, reward_combination=3, render=False)
        print(f"Hyperparameter Set {idx}:")
        print(f"  Average Rewards: {avg_rewards}")
        print(f"  Average Timesteps: {avg_timesteps}")
        print(f"  Average Penalties: {avg_penalties}")
        print("\n")
