import numpy as np
from typing import Tuple
import unittest
from scipy.ndimage import gaussian_filter


class Environment:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.player_x = np.random.randint(width)
        self.player_y = np.random.randint(height)

        self.fruit_x = np.random.randint(width)
        self.fruit_y = np.random.randint(height)

        self.state = np.zeros((2, self.width, self.height))

        # Define action space (up, down, left, right)
        self.num_actions = 4

        # Indicates whether the environment has been solved
        self.done = False

        self.env_name = "fruit"

        self.reset()

    def step(self, action: int, numeric: bool = False) -> Tuple[np.ndarray, float, bool]:

        self.state[0][self.player_x][self.player_y] = 0
        self.state[1][self.fruit_x][self.fruit_y] = 0

        # up 
        if action == 0:
            self.player_y = max(0, self.player_y - 1)
        # down
        elif action == 1:
            self.player_y = min(self.height - 1, self.player_y + 1)
        # left
        elif action == 2:
            self.player_x = max(0, self.player_x - 1)
        # right
        elif action == 3:
            self.player_x = min(self.width - 1, self.player_x + 1)

        self.state[0][self.player_x][self.player_y] = 1
        self.state[1][self.fruit_x][self.fruit_y] = 1

        self.done = ((self.player_x == self.fruit_x) and (self.player_y == self.fruit_y))

        if numeric:
            return self._get_state_numeric_continuous_norm(), self._get_reward(), self.done
        else:
            return self._get_fuzzy_state(), self._get_reward(), self.done

    def reset(self, numeric: bool = False) -> np.ndarray:
        self.player_x = np.random.randint(self.width)
        self.player_y = np.random.randint(self.height)

        self.fruit_x = np.random.randint(self.width)
        self.fruit_y = np.random.randint(self.height)

        while self.fruit_x == self.player_x:
            self.fruit_x = np.random.randint(self.width)

        while self.fruit_y == self.player_y:
            self.fruit_y = np.random.randint(self.height)

        self.done = False
        self.state = np.zeros((2, self.width, self.height))

        self.state[0][self.player_x][self.player_y] = 1
        self.state[1][self.fruit_x][self.fruit_y] = 1

        if numeric:
            return self._get_state_numeric_continuous_norm()
        else:
            return self._get_fuzzy_state()

    def _get_state_numeric_continuous_norm(self) -> np.ndarray:
        return np.array([(self.player_x + 1)/ self.width,
                         (self.player_y + 1) / self.height,
                         (self.fruit_x + 1) / self.width,
                         (self.fruit_y + 1) / self.height])

    def _get_state_numeric_continuous(self) -> np.ndarray:
        return np.array([self.player_x, self.player_y, self.fruit_x, self.fruit_y])
    
    def _get_fuzzy_state(self) -> np.ndarray:
        fuzzy_state = np.zeros((2, self.width, self.height))
        fuzzy_state[0] = gaussian_filter(self.state[0], sigma=1, truncate=1.5, mode='constant')
        fuzzy_state[1] = gaussian_filter(self.state[1], sigma=1, truncate=1.5, mode='constant')

        fuzzy_state = (fuzzy_state - np.min(fuzzy_state)) / (np.max(fuzzy_state) - np.min(fuzzy_state))

        return np.around(fuzzy_state, 2)

    def _get_state_image(self) -> np.ndarray:
        return self.state
        
    def _get_reward(self) -> float:
        reward = -(abs(self.player_x - self.fruit_x) / (self.width - 1)
                   + abs(self.player_y - self.fruit_y) / (self.height - 1))
        return reward

    def _get_reward_sparse2(self) -> float:
        reward = (self.player_x == self.fruit_x) and (self.player_y == self.fruit_y)
        return reward

    def _get_reward_sparse(self) -> float:
        reward = 0 + ((self.player_x == self.fruit_x) and (self.player_y == self.fruit_y))
        return reward


class TestEnvironment(unittest.TestCase):
    def test_loop(self):
        env = Environment(width=21, height=21)
        for i in range(1000):
            action = np.random.randint(4)
            state_numeric, reward, done = env.step(action=action, numeric = False)
            print(env._get_fuzzy_state())
            env.reset(numeric=False)
