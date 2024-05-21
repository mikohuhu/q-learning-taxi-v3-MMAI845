import gym
import numpy as np
import pickle

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_combination):
        super(CustomRewardWrapper, self).__init__(env)
        self.reward_combination = reward_combination

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.reward_combination == 1:
            # Combination 1: High positive rewards, moderate penalties
            if reward == -10:  # illegal move
                reward = -5
            elif reward == 20:  # successful drop-off
                reward = 50
            elif reward == -1:  # every timestep
                reward = -1
        elif self.reward_combination == 2:
            # Combination 2: High penalties, moderate rewards
            if reward == -10:  # illegal move
                reward = -20
            elif reward == 20:  # successful drop-off
                reward = 30
            elif reward == -1:  # every timestep
                reward = -1
        elif self.reward_combination == 3:
            # Combination 3: Original rewards and penalties
            if reward == -10:  # illegal move
                reward = -10
            elif reward == 20:  # successful drop-off
                reward = 20
            elif reward == -1:  # every timestep
                reward = -1

        return state, reward, done, info

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

def evaluate_original(env_name, q_table, episodes, render=False):
    env = gym.make(env_name)
    avg_rewards, avg_timesteps, avg_penalties = evaluate_q_learning(env, q_table, episodes, render)
    env.close()
    return avg_rewards, avg_timesteps, avg_penalties

def run_evaluation(env_name, q_table, episodes, render=False):
    results = []
    for combination in range(1, 4):
        env = CustomRewardWrapper(gym.make(env_name), reward_combination=combination)
        avg_rewards, avg_timesteps, avg_penalties = evaluate_q_learning(env, q_table, episodes, render)
        results.append((combination, avg_rewards, avg_timesteps, avg_penalties))
        env.close()
    
    return results

if __name__ == "__main__":
    env_name = "Taxi-v3"
    episodes = 100  # Number of episodes for evaluation

    # Load the trained Q-table
    with open('q_table.pickle', 'rb') as f:
        q_table = pickle.load(f)

    print("Evaluating Original Environment:")
    orig_rewards, orig_timesteps, orig_penalties = evaluate_original(env_name, q_table, episodes, render=False)
    print(f"Original Environment:")
    print(f"  Average Rewards: {orig_rewards}")
    print(f"  Average Timesteps: {orig_timesteps}")
    print(f"  Average Penalties: {orig_penalties}")

    results = run_evaluation(env_name, q_table, episodes, render=False)

    for combination, avg_rewards, avg_timesteps, avg_penalties in results:
        print(f"Combination {combination}:")
        print(f"  Average Rewards: {avg_rewards}")
        print(f"  Average Timesteps: {avg_timesteps}")
        print(f"  Average Penalties: {avg_penalties}")
