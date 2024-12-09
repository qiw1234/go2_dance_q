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
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,  #3
                                             self.base_ang_vel * self.obs_scales.ang_vel,  #3
                                             self.projected_gravity,  #3
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  #18
                                             self.dof_vel * self.obs_scales.dof_vel,  #18
                                             self.actions,  #18
                                             self.frames  #68
                                             ), dim=-1)
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,  #3   #3
                                  self.projected_gravity,  #3   #3
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  #12   #18
                                  self.dof_vel * self.obs_scales.dof_vel,  #12  #18
                                  self.actions,  #12  #18
                                  ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # print(self.obs_buf) #use in debug

    def _create_envs(self):
        super()._create_envs()
        # get arm_indices
        arm_name = self.cfg.asset.arm_name
        self.arm_indices = torch.zeros(len(arm_name), dtype=torch.long, device=self.device, requires_grad=False)
        self.arm_indices = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], arm_name)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        if not self.cfg.env.RSI:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
        else:
            # 用于初始化
            traj_idxs = self.motion_loader.weighted_traj_idx_sample_batch(len(env_ids))
            init_times = np.zeros(len(env_ids), dtype=int)
            frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, init_times)
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

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
        arm_pos = self.rb_states[:, self.arm_indices, 0:3].view(self.num_envs, -1) - self.base_pos
        temp = torch.exp(-100 * torch.sum(torch.square(self.frames[:, 49:52] - arm_pos), dim=1))
        return temp

    def _reward_track_arm_rot(self):
        arm_rot = self.rb_states[:, self.arm_indices, 3:7].view(self.num_envs, -1)
        return torch.exp(-10 * torch.sum(torch.square(self.frames[:, 52:56] - arm_rot), dim=1))

    def _reward_survival(self):
        "如果这个回合存活，给一个奖励"
        survival_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) < 1.,
                                 dim=1)
        # print(survival_buf)
        return survival_buf

