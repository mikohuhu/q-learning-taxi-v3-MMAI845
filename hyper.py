import gym
import numpy as np
import pickle
import random

def set_seed(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

def train_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, render=False, seed=None):
    if seed is not None:
        set_seed(env, seed)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    all_rewards = []
    all_penalties = []
    all_timesteps = []

    for episode in range(episodes):
        state = env.reset()
        rewards, penalties, timesteps = 0, 0, 0
        done = False

        while not done:
            if render:
                env.render()

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, _ = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            rewards += reward
            state = next_state
            timesteps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        all_rewards.append(rewards)
        all_penalties.append(penalties)
        all_timesteps.append(timesteps)

    return q_table, all_rewards, all_penalties, all_timesteps

def evaluate_q_learning(env, q_table, episodes, render=False, seed=None):
    if seed is not None:
        set_seed(env, seed)
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

def run_evaluation_with_hyperparameters(env_name, episodes, hyperparams, render=False, seed=None):
    results = []
    for params in hyperparams:
        alpha, gamma, epsilon, epsilon_decay, epsilon_min = params
        env = gym.make(env_name)
        q_table, _, _, _ = train_q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min, render, seed)
        avg_rewards, avg_timesteps, avg_penalties = evaluate_q_learning(env, q_table, episodes, render, seed)
        results.append((params, avg_rewards, avg_timesteps, avg_penalties))
        env.close()
    
    return results

if __name__ == "__main__":
    env_name = "Taxi-v3"
    episodes = 1000  # Number of episodes for training
    eval_episodes = 100  # Number of episodes for evaluation
    seed = 42  # Setting a random seed for reproducibility

    # Define different hyperparameters to test
    hyperparams = [
        (0.1, 0.99, 1.0, 0.995, 0.1),  # Set 1 (Original Parameters)
        (0.5, 0.99, 1.0, 0.995, 0.1),  # Set 2
        (0.1, 0.9, 1.0, 0.995, 0.1),   # Set 3
        (0.1, 0.99, 1.0, 0.9, 0.1),    # Set 4
        (0.1, 0.99, 0.5, 0.995, 0.1),  # Set 5
    ]

    results = run_evaluation_with_hyperparameters(env_name, episodes, hyperparams, render=False, seed=seed)

    for params, avg_rewards, avg_timesteps, avg_penalties in results:
        print(f"Hyperparameters {params}:")
        print(f"  Average Rewards: {avg_rewards}")
        print(f"  Average Timesteps: {avg_timesteps}")
        print(f"  Average Penalties: {avg_penalties}")
