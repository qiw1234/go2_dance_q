import matplotlib.pyplot as plt
import numpy as np
# raisim
raisim_obs_path = 'BJ_Raisim/net/HSW/data/actor_state.csv'
# raisim_obs_path = 'BJ_Raisim/net/HSW/data/actor_state_TsingHua.csv'
raisim_torque_path = 'BJ_Raisim/net/HSW/data/torques.csv'
raisim_base_euler_path = 'BJ_Raisim/net/HSW/data/base_euler.csv'

# isaac gym
# wave
# isaacgym_obs_path = 'BJ_Raisim/net/HSW/data/wave_obs.csv'
# isaacgym_torque_path = 'BJ_Raisim/net/HSW/data/wave_torque.csv'
# swing
isaacgym_obs_path = 'BJ_Raisim/net/HSW/data/swing_obs.csv'
isaacgym_torque_path = 'BJ_Raisim/net/HSW/data/swing_torque.csv'
isaacgym_base_euler_path = 'BJ_Raisim/net/HSW/data/swing_base_euler.csv'
# trot
# isaacgym_obs_path = 'BJ_Raisim/net/HSW/data/trot_obs.csv'
# isaacgym_torque_path = 'BJ_Raisim/net/HSW/data/trot_torque.csv'


plot_raisim = True
plot_isaacgym = True

start = 0
end = 100

default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1., -1.5, -0.1, 1., -1.5, 0, 0, 0, 0, 0, 0]

raisim_obs = np.loadtxt(raisim_obs_path, delimiter=",")
isaacgym_obs = np.loadtxt(isaacgym_obs_path, delimiter=",")
isaacgym_torques = np.loadtxt(isaacgym_torque_path, delimiter=",")
raisim_torques = np.loadtxt(raisim_torque_path, delimiter=",")
isaacgym_base_euler = np.loadtxt(isaacgym_base_euler_path, delimiter=',')
raism_base_euler = np.loadtxt(raisim_base_euler_path, delimiter=',')


isaacgym_torques = isaacgym_torques[start:end, :]
raisim_torques = raisim_torques[start:end, :]

raisim_ang_vel = raisim_obs[start:end, 0:3]
raisim_projected_gravity = raisim_obs[start:end, 3: 6]
raisim_dof_pos = raisim_obs[start:end, 6:24]
raisim_actions = raisim_obs[start:end, 24:42]

isaacgym_ang_vel = isaacgym_obs[start:end, 0:3]
isaacgym_projected_gravity = isaacgym_obs[start:end, 3: 6]
isaacgym_dof_pos = isaacgym_obs[start:end, 6:24]
isaacgym_actions = isaacgym_obs[start:end, 24:42]

nb_rows = 3
nb_cols = 3
plt.rcParams['font.size'] = 20
fig, axs = plt.subplots(1, 2)

# 机身角速度
a = axs[0,]
if plot_raisim:
    a.plot(raisim_ang_vel[:, 0], label='raisim_ang_vel_x', c='r')
    a.plot(raisim_ang_vel[:, 1], label='raisim_ang_vel_y', c='g')
    a.plot(raisim_ang_vel[:, 2], label='raisim_ang_vel_z', c='b')
if plot_isaacgym:
    a.plot(isaacgym_ang_vel[:, 0], label='isaacgym_ang_vel_x', linestyle='--', c='r')
    a.plot(isaacgym_ang_vel[:, 1], label='isaacgym_ang_vel_y', linestyle='--', c='g')
    a.plot(isaacgym_ang_vel[:, 2], label='isaacgym_ang_vel_z', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='base ang vel')
a.legend()
# 投影重力
a = axs[1,]
if plot_raisim:
    a.plot(raisim_projected_gravity[:, 0], label='raisim_g_x', c='r')
    a.plot(raisim_projected_gravity[:, 1], label='raisim_g_y', c='g')
    a.plot(raisim_projected_gravity[:, 2], label='raisim_g_z', c='b')
if plot_isaacgym:
    a.plot(isaacgym_projected_gravity[:, 0], label='isaacgym_g_x', linestyle='--', c='r')
    a.plot(isaacgym_projected_gravity[:, 1], label='isaacgym_g_y', linestyle='--', c='g')
    a.plot(isaacgym_projected_gravity[:, 2], label='isaacgym_g_z', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='projected gravity')
a.legend()
# 关节角度
fig2, axs2 = plt.subplots(2, 2)
a = axs2[0, 0]
if plot_raisim:
    a.plot(raisim_dof_pos[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raisim_dof_pos[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raisim_dof_pos[:, 2], label='raisim_LF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 0], label='isaacgym_LF_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 1], label='isaacgym_LF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 2], label='isaacgym_LF_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs2[0, 1]
if plot_raisim:
    a.plot(raisim_dof_pos[:, 3], label='raisim_RF_hip', c='r')
    a.plot(raisim_dof_pos[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(raisim_dof_pos[:, 5], label='raisim_RF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 3], label='isaacgym_RF_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 4], label='isaacgym_RF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 5], label='isaacgym_RF_calf', linestyle='--', c='b')
    plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs2[1, 0]
if plot_raisim:
    a.plot(raisim_dof_pos[:, 6], label='raisim_LH_hip', c='r')
    a.plot(raisim_dof_pos[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(raisim_dof_pos[:, 8], label='raisim_LH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 6], label='isaacgym_LH_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 7], label='isaacgym_LH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 8], label='isaacgym_LH_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs2[1, 1]
if plot_raisim:
    a.plot(raisim_dof_pos[:, 9], label='raisim_LH_hip', c='r')
    a.plot(raisim_dof_pos[:, 10], label='raisim_LH_thigh', c='g')
    a.plot(raisim_dof_pos[:, 11], label='raisim_LH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 9], label='isaacgym_LH_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 10], label='isaacgym_LH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 11], label='isaacgym_LH_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

# 关节角速度
# a = axs[2, 0]
# a.plot(raisim_dof_pos[:, 24], label='raisim_LF_hip', c='r')
# a.plot(isaacgym_dof_pos[:, 24], label='isaacgym_LF_hip', linestyle='--', c='r')
# a.plot(raisim_dof_pos[:, 25], label='raisim_LF_thigh', c='g')
# a.plot(isaacgym_dof_pos[:, 25], label='isaacgym_LF_thigh', linestyle='--', c='g')
# a.plot(raisim_dof_pos[:, 26], label='raisim_LF_calf', c='b')
# a.plot(isaacgym_dof_pos[:, 26], label='isaacgym_LF_calf', linestyle='--', c='b')
# plt.rcParams['xtick.labelsize'] = 20
# a.set(title='dof vel')
# a.legend()
#
# a = axs[2, 1]
# a.plot(raisim_dof_pos[:, 27], label='raisim_RF_hip', c='r')
# a.plot(isaacgym_dof_pos[:, 27], label='isaacgym_RF_hip', linestyle='--', c='r')
# a.plot(raisim_dof_pos[:, 28], label='raisim_RF_thigh', c='g')
# a.plot(isaacgym_dof_pos[:, 28], label='isaacgym_RF_thigh', linestyle='--', c='g')
# a.plot(raisim_dof_pos[:, 29], label='raisim_RF_calf', c='b')
# a.plot(isaacgym_dof_pos[:, 29], label='isaacgym_RF_calf', linestyle='--', c='b')
# plt.rcParams['xtick.labelsize'] = 20
# a.set(title='dof vel')
# a.legend()

# last_action
fig3, axs3 = plt.subplots(2, 2)
a = axs3[0, 0]
if plot_raisim:
    a.plot(raisim_actions[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raisim_actions[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raisim_actions[:, 2], label='raisim_LF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 0], label='isaacgym_LF_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 1], label='isaacgym_LF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 2], label='isaacgym_LF_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')

a = axs3[0, 1]
if plot_raisim:
    a.plot(raisim_actions[:, 3], label='raisim_RF_hip', c='r')
    a.plot(raisim_actions[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(raisim_actions[:, 5], label='raisim_RF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 3], label='isaacgym_RF_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 4], label='isaacgym_RF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 5], label='isaacgym_RF_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')

a = axs3[1, 0]
if plot_raisim:
    a.plot(raisim_actions[:, 6], label='raisim_LH_hip', c='r')
    a.plot(raisim_actions[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(raisim_actions[:, 8], label='raisim_LH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 6], label='isaacgym_LH_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 7], label='isaacgym_LH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 8], label='isaacgym_LH_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')

a = axs3[1, 1]
if plot_raisim:
    a.plot(raisim_actions[:, 9], label='raisim_RH_hip', c='r')
    a.plot(raisim_actions[:, 10], label='raisim_RH_thigh', c='g')
    a.plot(raisim_actions[:, 11], label='raisim_RH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 9], label='isaacgym_RH_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 10], label='isaacgym_RH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 11], label='isaacgym_RH_calf', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')
# plt.show()

# # torque
# fig4, axs4 = plt.subplots(2, 2)
# a = axs4[0, 0]
# a.plot(raisim_torques[:, 0], label='raisim_LF_hip', c='r')
# a.plot(raisim_torques[:, 1], label='raisim_LF_thigh', c='g')
# a.plot(raisim_torques[:, 2], label='raisim_LF_calf', c='b')
# a.plot(isaacgym_torques[:, 0], label='isaacgym_LF_hip', linestyle='--', c='r')
# a.plot(isaacgym_torques[:, 1], label='isaacgym_LF_thigh', linestyle='--', c='g')
# a.plot(isaacgym_torques[:, 2], label='isaacgym_LF_calf', linestyle='--', c='b')
# plt.rcParams['xtick.labelsize'] = 20
# a.set(title='torque')
#
# a = axs4[0, 1]
# a.plot(raisim_torques[:, 3], label='raisim_RF_hip', c='r')
# a.plot(raisim_torques[:, 4], label='raisim_RF_thigh', c='g')
# a.plot(raisim_torques[:, 5], label='raisim_RF_calf', c='b')
# a.plot(isaacgym_torques[:, 3], label='isaacgym_RF_hip', linestyle='--', c='r')
# a.plot(isaacgym_torques[:, 4], label='isaacgym_RF_thigh', linestyle='--', c='g')
# a.plot(isaacgym_torques[:, 5], label='isaacgym_RF_calf', linestyle='--', c='b')
# plt.rcParams['xtick.labelsize'] = 20
# a.set(title='torque')
#
# a = axs4[1, 0]
# a.plot(raisim_torques[:, 6], label='raisim_LH_hip', c='r')
# a.plot(raisim_torques[:, 7], label='raisim_LH_thigh', c='g')
# a.plot(raisim_torques[:, 8], label='raisim_LH_calf', c='b')
# a.plot(isaacgym_torques[:, 6], label='isaacgym_LH_hip', linestyle='--', c='r')
# a.plot(isaacgym_torques[:, 7], label='isaacgym_LH_thigh', linestyle='--', c='g')
# a.plot(isaacgym_torques[:, 8], label='isaacgym_LH_calf', linestyle='--', c='b')
# plt.rcParams['xtick.labelsize'] = 20
# a.set(title='torque')
#
# a = axs4[1, 1]
# a.plot(raisim_torques[:, 9], label='raisim_RH_hip', c='r')
# a.plot(raisim_torques[:, 10], label='raisim_RH_thigh', c='g')
# a.plot(raisim_torques[:, 11], label='raisim_RH_calf', c='b')
# a.plot(isaacgym_torques[:, 9], label='isaacgym_RH_hip', linestyle='--', c='r')
# a.plot(isaacgym_torques[:, 10], label='isaacgym_RH_thigh', linestyle='--', c='g')
# a.plot(isaacgym_torques[:, 11], label='isaacgym_RH_calf', linestyle='--', c='b')
# plt.rcParams['xtick.labelsize'] = 20
# a.set(title='torque')

# plot base_euler
fig5, axs5 = plt.subplots()
a = axs5
if plot_raisim:
    a.plot(raism_base_euler[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raism_base_euler[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raism_base_euler[:, 2], label='raisim_LF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_base_euler[:, 0], label='isaacgym_base_x', linestyle='--', c='r')
    a.plot(isaacgym_base_euler[:, 1], label='isaacgym_base_y', linestyle='--', c='g')
    a.plot(isaacgym_base_euler[:, 2], label='isaacgym_base_z', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='base euler')

plt.show()
