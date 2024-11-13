import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.data.types import Trajectory, TrajectoryWithRew
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import matplotlib.pyplot as plt

def generate_acts(stepNum):
    acts = []
    y_real = []
    y_compute = []
    act = 0
    y_pre = 0
    for i in range(stepNum):
        y = np.sin( i / (stepNum-1) * np.pi)
        # if i != 0:
        #     act = np.cos(i / stepNum * np.pi) / stepNum * np.pi
        # else:
        #     act = 0
        act = np.cos(i / (stepNum-1) * np.pi) / (stepNum-1) * np.pi
        y_pre = act + y_pre
        y_compute.append(y_pre)
        acts.append([act])
        y_real.append(y)
    return acts, y_real, y_compute

def generate_trajectories(acts_val,env):
    obs_val = []
    rews_val = []
    dones_val = []
    infos_val = []
    # obs_temp,_ = env.reset()
    obs_temp, _ = env.reset()
    # print(acts_val,env)
    obs_val.append(obs_temp)
    for act_temp in acts_val:
        obs_temp, rews_temp, dones_temp, _, infos_temp = env.step([act_temp])# env
        # obs_temp, rews_temp, dones_temp, infos_temp = env.step([act_temp])# venv
        # print(act_temp, obs_temp, rews_temp, dones_temp, infos_temp)
        obs_val.append(list(obs_temp))
        rews_val.append(float(rews_temp))
        infos_val.append(infos_temp)
    dones_val.append(dones_temp)

    print(acts_val, np.array(obs_val), np.array(rews_val), np.array(dones_val), np.array(infos_val))
    roll_out = []
    roll_out = Trajectory(obs=np.array(obs_val),
                          acts=acts_val,
                          infos=np.array(infos_val),
                          terminal=True)
    # roll_out_rw = TrajectoryWithRew(roll_out,rews=np.array(rews_val))
    roll_out_rw = TrajectoryWithRew(obs=np.array(obs_val),
                                    rews=np.array(rews_val),
                                    acts=acts_val,
                                    infos=None,
                                    terminal=True)
    print(type(roll_out_rw))
    return roll_out_rw,roll_out
def generate_rollouts(env):
    stepNum = 100
    acts, y_real, y_compute = generate_acts(stepNum)
    acts_val = np.array(acts)
    roll_out_rw, _ = generate_trajectories(acts_val, env)
    return roll_out_rw

stepNum = 100
acts, y_real, y_compute = generate_acts(stepNum)
print('expert_act_error:',y_real[99]-y_compute[99])
x =range(stepNum)
line1, = plt.plot(x,y_real,'--*r')
line2, = plt.plot(x,y_compute,'-.+g')
plt.legend((line1,line2),('y_real','y_compute'))
# plt.show()



SEED = 0
# Option A: use the `make_vec_env` helper function - make sure to pass `post_wrappers=[lambda env, _: RolloutInfoWrapper(env)]`
env = gym.make('trackingEnv.env.TrackingEnv:TrackEnv-v0')
venv = make_vec_env(
    "trackingEnv.env.TrackingEnv:TrackEnv-v0",
    rng=np.random.default_rng(),# 作为随机种子作用于环境
    n_envs=1,# 复制环境个数
    parallel = False,# 是否开启并行计算（子线程）
    post_wrappers=[lambda venv, _: RolloutInfoWrapper(venv)],
)

venv.reset()


rollouts = generate_rollouts(env)
rollouts = [rollouts]
print(rollouts)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=1024,
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
    demo_batch_size=100,# num of trajecotries points
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
learner.save('learner')

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))
