import numpy as np
import pickle
from custom_taxi_env import CustomTaxiEnv  # Import the custom environment
import time

def evaluate_single_qtable(q_table, env, num_trials, max_steps=100):
    total_epochs, total_penalties, total_rewards = 0, 0, 0

    for trial in range(num_trials):
        state = env.reset()
        epochs, num_penalties, rewards = 0, 0, 0

        print(f"EPISODE {trial + 1}:\n")
        env.render()

        while True:
            action = np.argmax(q_table[state])
            state, reward, done, _ = env.step(action)

            if reward == -10:
                num_penalties += 1

            rewards += reward
            epochs += 1

            print(f"TRAINED AGENT\n")
            print("Step {}\n".format(epochs))
            env.render()
            print(f"score: {rewards}\n")
            time.sleep(1)  # Slow down the rendering for better visualization

            if done or epochs >= max_steps:
                break

        total_penalties += num_penalties
        total_epochs += epochs
        total_rewards += rewards

    average_time = total_epochs / float(num_trials)
    average_penalties = total_penalties / float(num_trials)
    average_rewards = total_rewards / float(num_trials)

    return average_time, average_rewards, average_penalties

if __name__ == "__main__":
    env = CustomTaxiEnv()  # Use the custom environment
    num_trials = 10  # Number of evaluation trials per Q-table
    max_steps = 100  # Maximum steps per episode

    # Load the Q-table with the specified hyperparameters
    filename = "q_table_2.pickle"
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)

    average_time, average_rewards, average_penalties = evaluate_single_qtable(q_table, env, num_trials, max_steps)

    print("Evaluation complete:")
    print(f"Average Time Steps: {average_time}")
    print(f"Average Rewards: {average_rewards}")
    print(f"Average Penalties: {average_penalties}")

    # Watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    print("Watching trained agent...\n")

    for s in range(max_steps):
        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        time.sleep(1)  # Slow down the rendering for better visualization

        if done:
            break

    env.close()
