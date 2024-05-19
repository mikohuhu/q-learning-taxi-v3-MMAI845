import gym
from gym import spaces, logger
import numpy as np

class CustomTaxiEnv(gym.Env):
    def __init__(self):
        super(CustomTaxiEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # 6 possible actions: 4 movements, pickup, dropoff
        self.observation_space = spaces.Discrete(5000)  # Simplified state space for demo
        
        self.passenger_locations = [(0, 0), (0, 4), (4, 0), (4, 3), (2, 2)]  # Extra passenger locations
        self.destination = (4, 4)
        self.walls = [(1, 2), (2, 2), (3, 2)]  # Example walls

        self.state = None
        self.taxi_position = None
        self.passenger_index = None

    def reset(self):
        while True:
            self.taxi_position = (np.random.randint(5), np.random.randint(5))
            if self.taxi_position not in self.walls:
                break
                
        self.passenger_index = np.random.randint(len(self.passenger_locations))
        self.state = (self.taxi_position, self.passenger_locations[self.passenger_index])
        return self.encode(self.state)
    
    def encode(self, state):
        # Simplified encoding for demonstration
        taxi_row, taxi_col = state[0]
        pass_loc_index = self.passenger_locations.index(state[1])
        return taxi_row * 100 + taxi_col * 20 + pass_loc_index
    
    def step(self, action):
        taxi_row, taxi_col = self.taxi_position
        new_row, new_col = taxi_row, taxi_col

        if action == 0:  # move north
            new_row = max(taxi_row - 1, 0)
        elif action == 1:  # move south
            new_row = min(taxi_row + 1, 4)
        elif action == 2:  # move east
            new_col = min(taxi_col + 1, 4)
        elif action == 3:  # move west
            new_col = max(taxi_col - 1, 0)
        
        if (new_row, new_col) not in self.walls:
            self.taxi_position = (new_row, new_col)
        
        reward = -1  # Default reward, can be adjusted
        done = False
        
        if self.taxi_position == self.destination:
            reward = 20  # Reward for reaching the destination
            done = True
            
        self.state = (self.taxi_position, self.passenger_locations[self.passenger_index])
        return self.encode(self.state), reward, done, {}
    
    def render(self, mode='human'):
        out = np.array([['.' for _ in range(5)] for _ in range(5)])
        out[tuple(zip(*self.walls))] = 'W'  # Mark walls
        taxi_row, taxi_col = self.taxi_position
        out[taxi_row][taxi_col] = 'T'  # Mark taxi
        for i, loc in enumerate(self.passenger_locations):
            if i == self.passenger_index:
                out[loc] = 'P'  # Mark passenger
            # Destination can be marked differently if needed
        print("\n".join(["".join(row) for row in out]))
        
# Register the custom environment (do this outside the class, in your script or a main block)
gym.envs.registration.register(
    id='CustomTaxi-v0',
    entry_point='custom_taxi_env:CustomTaxiEnv',
)