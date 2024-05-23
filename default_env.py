import gym

# Create the Taxi-v3 environment
env = gym.make('Taxi-v3')

# Reset the environment to its initial state and get the initial observation
observation = env.reset()

# Render the initial state of the environment
env.render()

# Run a few steps in the environment
for _ in range(5):
    # Randomly sample an action from the action space
    action = env.action_space.sample()

    # Take a step in the environment with the sampled action
    observation, reward, done, info = env.step(action)

    # Render the updated state of the environment
    env.render()

    # Print the observation, reward, and whether the episode is done
    print('Observation:', observation)
    print('Reward:', reward)
    print('Done:', done)
    print('Info:', info)

    # Check if the episode is done
    if done:
        break

# Close the environment
env.close()
