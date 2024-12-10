from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_dance_config import GO2DanceCfg_beat
from legged_gym.motion_loader.motion_loader_panda_fixed_gripper import motionLoaderPandaFixedGripper
from legged_gym.envs.base.legged_robot import LeggedRobot

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LeggedRobotPandaFixedGripper(LeggedRobot):
    def __init__(self, cfg: GO2DanceCfg_beat, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # 重新加载动作数据
        self.motion_loader = motionLoaderPandaFixedGripper(motion_files=self.cfg.env.motion_files, device=self.device,
                                                           time_between_frames=self.dt,
                                                           frame_duration=self.cfg.env.frame_duration)
        self.max_episode_length_s = self.motion_loader.trajectory_lens[0]
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)


    def compute_observations(self):
        """ Computes observations
        """
        # base: pos quat lin_vel ang_vel
        base_pos_error = self.base_pos - self.env_origins - self.frames[:, 0:3]
        base_euler_error = get_euler_xyz_tensor(self.base_quat) - get_euler_xyz_tensor(self.frames[:, 3:7])
        base_lin_vel_error = self.base_lin_vel - quat_rotate_inverse(self.frames[:, 3:7], self.frames[:, 7:10])
        base_lin_ang_error = self.base_ang_vel - quat_rotate_inverse(self.frames[:, 3:7], self.frames[:, 10:13])
        # foot: pos q dq
        foot_pos_error = self.toe_pos_body - self.frames[:, 13:25]
        leg_dof_pos_error = self.dof_pos[:, 0:12] - self.frames[:, 25:37]  # LF RF LH RH
        leg_dof_vel_error = self.dof_vel[:, 0:12] - self.frames[:, 37:49]
        # arm: pos quat q dq
        arm_end_pos_error = self.arm_end_pos - self.frames[:, 49:52]
        arm_end_rot_error = get_euler_xyz_tensor(self.arm_end_quat) - get_euler_xyz_tensor(self.frames[:, 52:56])
        arm_dof_pos_error = self.dof_pos[:, 12:18] - self.frames[:, 56:62]
        arm_dof_vel_error = self.dof_vel[:, 12:18] - self.frames[:, 62:68]

        tracking_error = torch.cat((base_pos_error, base_euler_error, base_lin_vel_error, base_lin_ang_error,
                                    foot_pos_error, leg_dof_pos_error, leg_dof_vel_error,
                                    arm_end_pos_error, arm_end_rot_error, arm_dof_pos_error, arm_dof_vel_error), dim=-1)

        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,  # 3   # 0...3
                                             self.base_ang_vel * self.obs_scales.ang_vel,  # 3   # 3...6
                                             self.projected_gravity,  # 3   # 6...9
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 18   # 9...27
                                             self.dof_vel * self.obs_scales.dof_vel,  # 18  # 27...45
                                             self.actions,  # 18  # 45...63
                                             self.base_euler_xyz * self.obs_scales.quat, # 3  63...66
                                             tracking_error  # 66   66...132
                                             ), dim=-1)
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,  # 3   # 3
                                  self.projected_gravity,  # 3   # 6
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 18   # 24
                                  self.dof_vel * self.obs_scales.dof_vel,  # 18  # 42
                                  self.actions,  # 18  # 60
                                  ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # print(self.obs_buf) # use in debug

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:42] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[42:60] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _create_envs(self):
        super()._create_envs()
        # get arm_indices
        arm_name = self.cfg.asset.arm_name
        self.arm_link6_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], arm_name)

    # def reset_idx(self, env_ids):
    #     """ Reset some environments.
    #         Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
    #         [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
    #         Logs episode info
    #         Resets some buffers
    #
    #     Args:
    #         env_ids (list[int]): List of environment ids which must be reset
    #     """
    #     if len(env_ids) == 0:
    #         return
    #     # update curriculum
    #     if self.cfg.terrain.curriculum:
    #         self._update_terrain_curriculum(env_ids)
    #     # avoid updating command curriculum at each step since the maximum command is common to all envs
    #     if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
    #         self.update_command_curriculum(env_ids)
    #
    #     # reset robot states
    #     if not self.cfg.env.RSI:
    #         self._reset_dofs(env_ids)
    #         self._reset_root_states(env_ids)
    #     else:
    #         # 用于初始化
    #         traj_idxs = self.motion_loader.weighted_traj_idx_sample_batch(len(env_ids))
    #         init_times = np.zeros(len(env_ids), dtype=int)
    #         frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, init_times)
    #         self._reset_dofs_amp(env_ids, frames)
    #         self._reset_root_states_amp(env_ids, frames)
    #
    #     self._resample_commands(env_ids)
    #
    #     # reset buffers
    #     self.last_actions[env_ids] = 0.
    #     self.last_dof_vel[env_ids] = 0.
    #     self.feet_air_time[env_ids] = 0.
    #     self.episode_length_buf[env_ids] = 0
    #     self.reset_buf[env_ids] = 1
    #     # fill extras
    #     self.extras["episode"] = {}
    #     for key in self.episode_sums.keys():
    #         self.extras["episode"]['rew_' + key] = torch.mean(
    #             self.episode_sums[key][env_ids]) / self.max_episode_length_s
    #         self.episode_sums[key][env_ids] = 0.
    #     # log additional curriculum info
    #     if self.cfg.terrain.curriculum:
    #         self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
    #     if self.cfg.commands.curriculum:
    #         self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
    #     # send timeout info to the algorithm
    #     if self.cfg.env.send_timeouts:
    #         self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        包含amp的函数在这里就是初始化的时候把机器人的位置和关节角度设置成参考动作的状态
        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = torch.cat((self.motion_loader.get_joint_pose_batch(frames),
                                           self.motion_loader.get_arm_joint_pos_batch(frames)), dim=1)
        self.dof_vel[env_ids] = torch.cat((self.motion_loader.get_joint_vel_batch(frames),
                                           self.motion_loader.get_arm_joint_vel_batch(frames)), dim=1)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reward_track_arm_dof_pos(self):
        return torch.exp(-0.1 * torch.sum(torch.square(self.frames[:, self.motion_loader.ARM_JOINT_POS_START_IDX:
                                                                      self.motion_loader.ARM_JOINT_POS_END_IDX] -
                                                       self.dof_pos[:, 12:18]), dim=1))

    def _reward_track_griper_dof_pos(self):
        return torch.exp(-1000 * torch.sum(torch.square(self.frames[:, 62:64] - self.dof_pos[:, 18:20]), dim=1))

    def _reward_track_arm_dof_vel(self):
        return torch.exp(-0.1 * torch.sum(torch.square(self.frames[:, self.motion_loader.ARM_JOINT_VEL_START_IDX:
                                                                      self.motion_loader.ARM_JOINT_VEL_END_IDX] -
                                                       self.dof_vel[:, 12:18]), dim=1))

    def _reward_track_arm_pos(self):
        temp = torch.exp(-100 * torch.sum(torch.square(self.frames[:, 49:52] - self.arm_end_pos), dim=1))
        return temp

    def _reward_track_arm_rot(self):
        return torch.exp(-10 * torch.sum(torch.square(get_euler_xyz_tensor(self.arm_end_quat) -
                                                      get_euler_xyz_tensor(self.frames[:, 52:56])), dim=1))

    def _reward_survival(self):
        "如果这个回合存活，给一个奖励"
        survival_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) < 1.,
                                 dim=1)
        # print(survival_buf)
        return survival_buf

