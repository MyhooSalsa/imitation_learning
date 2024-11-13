import gymnasium as gym
from imitation.algorithms.bc import BC
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies.serialize import policy_registry
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
# from TrackingEnv import TrackingEnv


from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from imitation.data import types

SEED = 0
# Option A: use the `make_vec_env` helper function - make sure to pass `post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]`
# env = gym.make('imitation_learning.trackingEnv.env.TrackingEnv:TrackEnv-v0')
venv = make_vec_env(
    "trackingEnv.env.TrackingEnv:TrackEnv-v0",
    rng=np.random.default_rng(),# 作为随机种子作用于环境
    n_envs=8,# 复制环境个数
    parallel = False,# 是否开启并行计算（子线程）
    post_wrappers=[lambda venv, _: RolloutInfoWrapper(venv)],
)

venv.reset()
# model = PPO('MlpPolicy', env)
expert = PPO(
    policy=MlpPolicy,
    env=venv,
    seed=42,
    batch_size=16,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=32,
)
expert.learn(10)  # Note: set to 100000 to train a proficient expert
# expert.save(f"{TIMESTEPS * iters}")

rng = np.random.default_rng()# 随机生成个数
# rollouts包含min_episodes个list类型的rollout，各个rollout类型为imitation.data.types.TrajectoryWithRew
# 包含一个回合中的np.array(obs),np.array(acts),np.array(rwds),str(infos)和bool(terminal)
# rollouts本意为通过专家策略生成专家轨迹，当存在专家轨迹时，可通过直接对rollouts赋值进行操作
rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)
print('GAIL样本：')
print(rollouts)
print('rollouts变量类型：')
print(type(rollouts[0]))
print('rollouts数量：')
print(len(rollouts))

for rollout in rollouts:
    print("----------\n",rollout)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
)
reward_net = BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

# evaluate the learner before training
venv.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
gail_trainer.train(800000)  # Train for 800_000 steps to match expert.
venv.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))
