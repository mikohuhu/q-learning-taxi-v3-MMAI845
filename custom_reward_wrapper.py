import gym

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