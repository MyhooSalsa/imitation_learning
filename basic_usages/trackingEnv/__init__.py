# from gymnasium.envs.registration import register
import gymnasium as gym
from basic_usages.trackingEnv.version import VERSION as __version__

gym.register(
    id="TrackEnv-v0",
    entry_point="basic_usages.trackingEnv.env.TrackingEnv:TrackEnv",
)
