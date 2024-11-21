import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl
import copy
import time

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def load_policy(logdir, actor_critic):
    # body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    # import os
    # adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    adaptation_module = actor_critic.adaptation_module
    body = actor_critic.actor_body

    jit_adaptation_module = copy.deepcopy(adaptation_module).to('cpu')
    traced_script_adaptation_module = torch.jit.script(jit_adaptation_module)
    traced_script_adaptation_module.save("adaptation_module_latest.jit")

    jit_body = copy.deepcopy(body).to('cpu')
    traced_script_body = torch.jit.script(jit_body)
    traced_script_body.save("body_latest.jit")

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"./runs/{label}/*")
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    actor_critic = ActorCritic(env.num_obs,
                                      env.num_privileged_obs,
                                      env.num_obs_history,
                                      env.num_actions,
                                      ).to("cpu")
    weights = torch.load(logdir + "/checkpoints/ac_weights_002000.pt")
    # weights = torch.load("pretrained/ac_weights_040000.pt")
    actor_critic.load_state_dict(state_dict=weights)

    policy = load_policy(logdir,actor_critic)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/2024-11-21/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 500
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    measured_y_vels = np.zeros(num_eval_steps)
    measured_ang_vels = np.zeros(num_eval_steps)
    measured_base_height = np.zeros(num_eval_steps)
    trajectory_x = np.zeros(num_eval_steps)
    trajectory_y = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    # print(obs)

    # for i in range(100):
    #     actions = torch.zeros(1, 12)
    #     # env.env.p_gains = 80.0
    #     # env.env.d_gains = 4.0
    #     obs, rew, done, info = env.step(actions)
    #     # print(env.root_states[0, 2])

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        # env.env.p_gains = 20.0
        # env.env.d_gains = 0.5
        env.commands[:, 0] = x_vel_cmd #*i/num_eval_steps
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_y_vels[i] = env.base_lin_vel[0, 1]
        measured_ang_vels[i] = env.base_ang_vel[0, 2]
        measured_base_height[i] = env.root_states[0, 2]
        trajectory_x[i] = env.root_states[0, 0]
        trajectory_y[i] = env.root_states[0, 1]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(5, 1, figsize=(12, 12))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y_vels, color='black', linestyle="-", label="Measured")
    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), np.zeros(num_eval_steps), color='black', linestyle="--", label="Desired")
    axs[2].set_title("Y Velocity")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Velocity (m/s)")

    axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_ang_vels, color='black', linestyle="-", label="Measured")
    axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), np.zeros(num_eval_steps), color='black', linestyle="--", label="Desired")
    axs[3].set_title("Ang Velocity")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Velocity (rad/s)")

    axs[4].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_base_height, color='black', linestyle="-", label="Measured")
    axs[4].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), np.ones(num_eval_steps)*0.3, color='black', linestyle="--", label="Desired")
    axs[4].set_title("Base Height")
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("m")

    plt.tight_layout()
    plt.show()

    plt.plot(trajectory_x, trajectory_y, color='black', linestyle="-", label="Measured")
    plt.title("Trajectory Visual")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
