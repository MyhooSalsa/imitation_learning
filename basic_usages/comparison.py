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
    roll_out_rw,_ = generate_trajectories(acts_val, env)
    return roll_out_rw

stepNum = 100
acts, y_real, y_compute = generate_acts(stepNum)
print('expert_act_error:',y_real[99]-y_compute[99])
x =range(stepNum)
line1, = plt.plot(x,y_real,'--*r')
line2, = plt.plot(x,y_compute,'-.+g')
plt.legend((line1,line2),('y_real','y_compute'))
# plt.show()
