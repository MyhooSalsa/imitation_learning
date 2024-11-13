import gymnasium as gym
import numpy as np


class TrackEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
    action_space = gym.spaces.Box(low=-.1, high=.1, shape=(1,), dtype=float)
    def __init__(self):
        super().__init__()
        self.y = 0
        self.stepNum = 0
        self.y_pre = 0
        self.error = 0
        self.max_iter = 4
        self.epsilon = 1e-3
    def compute_observation(self):
        state = np.zeros(2)
        self.error = np.sin(self.stepNum)-self.y
        self.y_pre = self.y
        state[0] = self.error
        state[1] = self.y_pre
        return state
    def is_done(self):
        if self.stepNum == self.max_iter and abs(self.error)<self.epsilon:
            return True
        else:
            return False
    def is_truncated(self):# 人为截断/停止

        if self.stepNum > self.max_iter:
            return True
        else:
            return False
    def compute_reward(self):
        return -abs(self.error)
    def step(self,action):
        self.stepNum += 1
        self.y = self.y + action[0]
        # print("stepNum:", self.stepNum)
        # print("y:", self.y)
        return self.compute_observation(), self.compute_reward(),self.is_done(), self.is_truncated(), {}
    def reset(self, seed=None, options=None):
        self.y = 0
        self.stepNum = 0
        self.y_pre = 0
        self.error = 0
        return self.compute_observation(), {}
    def render(self):
        pass
    def close(self):
        pass
    def connect(self, render):
        pass
    def seed(self):
        pass
