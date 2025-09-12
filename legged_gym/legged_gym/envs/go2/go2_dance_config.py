from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters: 调整为更接近真实硬件的参数
        control_type = 'P'
        stiffness = {'joint': 15.}  # 从20降低到15，更接近真实电机
        damping = {'joint': 0.3}    # 从0.5降低到0.3，减少阻尼
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "hip"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        max_contact_force = 90

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0.
            track_root_height = 0.5
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1.


    class env(LeggedRobotCfg.env):
        # 旋转跳跃参考动作
        motion_files = 'opti_traj/output_json'
        motion_name = None
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 12
        num_observations = 42
        num_privileged_obs = 96

class GO2Cfg_PPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_dance'
        resume_path = None

#-----------------------------swing----------------------------------------------------
class GO2DanceCfg_swing(GO2Cfg):
    class rewards( GO2Cfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales( GO2Cfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0.
            track_root_height = 0.5
            track_root_rot = 0
            orientation = -0.1
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 5
            track_dof_vel = 5
            track_toe_pos = 10.
            tracking_yaw = 10.

    class env(GO2Cfg.env):
        # 旋转跳跃参考动作
        motion_name = 'swing'

    class noise(GO2Cfg.noise):
        class noise_scales(GO2Cfg.noise.noise_scales):
            dof_pos = 0.01


class GO2DanceCfg_swingPPO( GO2Cfg_PPO ):
    class runner( GO2Cfg_PPO.runner ):
        experiment_name = 'go2_dance_swing'

#-----------------------------beat----------------------------------------------------
class GO2DanceCfg_beat(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0
            track_root_rot = 2.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 5.

    class env(GO2Cfg.env):
        motion_name = 'beat'

class GO2DanceCfg_beatPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_dance_beat'
        # resume_path = "legged_gym/log/GO2_new/keep_the_beat/model_1500.pt"

#-----------------------------jump----------------------------------------------------
class GO2DanceCfg_turn_and_jump(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        base_height_jump = 0.496

        class scales(GO2Cfg.rewards.scales):
            lin_vel_z = -0
            survival = 1
            feet_air_time = 1
            feet_contact_time = 0
            feet_contact_forces = -0.05
            action_rate = -0.3
            # arm_dof_error = -2.
            orientation = -10.
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 5
            track_root_height = 5
            track_root_rot = 12
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 0
            track_toe_height = 5
            track_toe_x = 5
            track_toe_y = 5
            # jump reward
            jump = 20.

    class env(GO2Cfg.env):
        motion_name = 'turn_and_jump'



class GO2DanceCfg_turn_and_jumpPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_dance_turn_and_jump'
        # resume_path = 'legged_gym/log/GO2_new/turn_and_jump/model_550.pt'




#-----------------------------wave----------------------------------------------------
class GO2DanceCfg_wave(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            action_rate = -1
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 5
            track_root_height = 0.
            track_root_rot = 5
            orientation = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 10
            track_dof_vel = 10
            track_toe_pos = 20 #20 5

    class env(GO2Cfg.env):
        motion_name = 'wave'


class GO2DanceCfg_wavePPO(GO2Cfg_PPO):

    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_wave'
        # resume_path = "legged_gym/log/GO2_new/wave/model_1500.pt"

#-----------------------------pace----------------------------------------------------
class GO2DanceCfg_pace(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0

            lin_vel_z = -0
            track_root_pos = 0.
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            termination = -1.0

    class env(GO2Cfg.env):
        motion_name = 'pace'




class GO2DanceCfg_pacePPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_pace'
        # resume_path = 'legged_gym/logs/go2_trot/Nov02_13-32-18_/model_1400.pt'
        # resume_path = "legged_gym/log/GO2_new/pace/model_1500.pt"

#-----------------------------trot----------------------------------------------------
class GO2DanceCfg_trot(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            feet_slip = -1

            lin_vel_z = -0
            track_root_pos = 0.
            track_root_rot = 1
            track_lin_vel_ref = 1
            track_ang_vel_ref = 1
            track_dof_pos = 5
            track_dof_vel = 1
            track_toe_pos = 10
            termination = -1.0

    class env(GO2Cfg.env):
        motion_name = 'trot'


class GO2DanceCfg_trotPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_trot'
        # resume_path = 'legged_gym/logs/go2_trot/Apr09_11-59-28_/model_123000.pt'
        resume_path = "legged_gym/logs/go2_trot/Apr14_13-17-43_/model_1500.pt"
#-----------------------------stand----------------------------------------------------
class GO2DanceCfg_stand(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            action_rate = -10

            orientation = -2.
            dof_vel = -0.001  # -0.01太大  -0.0001还行  -0.001可以 -0.002太大机械臂起不来
            dof_acc = -3e-5  # -2.5e-5太大起不来 -5e-7 -5e-6还行 -5e-5起不来 -1e-5还可以 -2e-5可以 -4e-5起不来
            feet_contact_forces = -0.01

            # 模仿奖励
            track_root_rollandpitch = 10.
            track_root_pos = 5
            track_root_height = 0
            track_root_rot = 5
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 5
            track_dof_vel = 0
            track_toe_pos = 5

    class env(GO2Cfg.env):
        motion_name = 'stand'
        # 站立参考轨迹
        motion_files = 'opti_traj/go2ST'
        # check_contact = False



class GO2DanceCfg_standPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_stand'
        resume_path = 'legged_gym/logs/go2_stand/Apr17_16-51-53_/model_60000.pt'

#-----------------------------sidestep----------------------------------------------------
class GO2DanceCfg_sidestep(GO2Cfg):
    class control(GO2Cfg.control):
        # 为sidestep任务调整PD参数，使其更接近真实硬件
        stiffness = {'joint': 12.}  # 进一步降低刚度
        damping = {'joint': 0.2}    # 进一步降低阻尼

    class domain_rand(GO2Cfg.domain_rand):
        # 增强域随机化以提高鲁棒性
        randomize_friction = True
        friction_range = [0.1, 2.5]  # 扩大摩擦力范围

        randomize_motor = True
        motor_strength_range = [0.7, 1.3]  # 扩大电机强度范围

        randomize_torque = True
        torque_multiplier_range = [0.7, 1.3]  # 扩大力矩范围

        randomize_base_mass = True
        added_mass_range = [-2., 8.]  # 扩大质量随机化范围

        randomize_base_com = True
        added_com_range = [-0.08, 0.08]  # 扩大质心随机化范围

        push_robots = True
        push_interval_s = 10  # 更频繁的推力
        max_push_vel_xy = 1.5  # 更大的推力

        # 启用动作延迟
        action_delay = True
        action_curr_step = [1, 3]  # 更大的延迟范围
        action_delay_range = [0.02, 0.08]  # 20-80ms延迟，与50Hz控制频率匹配

    class noise(GO2Cfg.noise):
        # 增强观测噪声
        class noise_scales(GO2Cfg.noise.noise_scales):
            dof_pos = 0.06  # 增加关节位置噪声
            dof_vel = 2.0   # 增加关节速度噪声
            lin_vel = 0.15  # 增加线速度噪声
            ang_vel = 0.3   # 增加角速度噪声
            gravity = 0.08  # 增加重力噪声

    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            # 参考 trot 的奖励结构 + 保留足端碰撞/接触惩罚
            # 基础惩罚（与 trot 对齐）
            torques = -0.0002
            dof_pos_limits = -10.0
            feet_slip = -1
            termination = -1.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0.

            # 保留：足端碰撞/接触惩罚
            collision = -5.0            # 碰撞惩罚
            feet_contact_forces = -0.02 # 接触力过大惩罚

            # 模仿/跟踪（以 trot 为基线，强化侧向速度）
            track_root_rot = 1
            track_lin_vel_ref = 2       # 侧移任务，略强于 trot 的线速度跟踪
            track_ang_vel_ref = 1
            track_dof_pos = 5
            track_dof_vel = 1
            track_toe_pos = 10

    class env(GO2Cfg.env):
        motion_name = 'sidestep_truncated'

class GO2DanceCfg_sidestepPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_sidestep'
        # resume_path = None  # 可以在训练后设置

# 训练脚本示例 (可以保存为单独的train_sidestep.py文件):
"""
训练sidestep模型的示例脚本:

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # 创建环境
    env = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # 创建PPO训练器
    ppo_runner = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 开始训练
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.task = 'go2_dance_sidestep'  # 使用新的sidestep配置
    train(args)
"""
