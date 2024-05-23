import gym
from gym import spaces
import numpy as np
import time

class CustomTaxiEnv(gym.Env):
    def __init__(self):
        super(CustomTaxiEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # 6 possible actions: 4 movements, pickup, dropoff
        self.observation_space = spaces.Discrete(10000)  # 10x10 grid, 5 passenger locations, simplified state space for demo

        self.passenger_locations = [(0, 0), (0, 9), (9, 0), (9, 8), (5, 5)]  # Extra passenger locations
        self.destination = (9, 9)
        self.walls = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4),  # Example walls
                      (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (5, 8), (5, 9)]

        self.state = None
        self.taxi_position = None
        self.passenger_index = None

    def reset(self):
        while True:
            self.taxi_position = (np.random.randint(10), np.random.randint(10))
            if self.taxi_position not in self.walls:
                break

        self.passenger_index = np.random.randint(len(self.passenger_locations))
        self.state = (self.taxi_position, self.passenger_index)
        return self.encode(self.state)

    def encode(self, state):
        taxi_row, taxi_col = state[0]
        pass_loc_index = state[1]
        return taxi_row * 100 + taxi_col * 10 + pass_loc_index

    def step(self, action):
        taxi_row, taxi_col = self.taxi_position
        new_row, new_col = taxi_row, taxi_col

        if action == 0:  # move north
            new_row = max(taxi_row - 1, 0)
        elif action == 1:  # move south
            new_row = min(taxi_row + 1, 9)
        elif action == 2:  # move east
            new_col = min(taxi_col + 1, 9)
        elif action == 3:  # move west
            new_col = max(taxi_col - 1, 0)

        if (new_row, new_col) not in self.walls:
            self.taxi_position = (new_row, new_col)

        reward = -1  # Default reward, can be adjusted
        done = False

        if self.taxi_position == self.destination:
            reward = 20  # Reward for reaching the destination
            done = True

        self.state = (self.taxi_position, self.passenger_index)
        return self.encode(self.state), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            out = np.array([[' ' for _ in range(10)] for _ in range(10)])
            for (i, j) in self.walls:
                out[i][j] = '|'

            taxi_row, taxi_col = self.taxi_position
            out[taxi_row][taxi_col] = 'T'

            pass_row, pass_col = self.passenger_locations[self.passenger_index]
            out[pass_row][pass_col] = 'P'

            dest_row, dest_col = self.destination
            out[dest_row][dest_col] = 'D'

            print("+---------+")
            for row in out:
                print("|" + "".join(row) + "|")
            print("+---------+")
            time.sleep(1)
        else:
            super(CustomTaxiEnv, self).render(mode=mode)  # Render using default behavior

# Register the custom environment (do this outside the class, in your script or a main block)
gym.envs.registration.register(
    id='CustomTaxi-v0',
    entry_point='custom_taxi_env:CustomTaxiEnv',
)
