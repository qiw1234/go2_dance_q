import matplotlib.pyplot as plt
import numpy as np

def quaternion_to_euler(quaternions):
    """
    将多组四元数 (x, y, z, w) 转换为多组欧拉角 (roll, pitch, yaw)。
    输入：quaternions - 二维数组，形状为 (N, 4)，每行是一个四元数 (x, y, z, w)。
    输出：欧拉角数组，形状为 (N, 3)，每行是一个欧拉角 (roll, pitch, yaw)。
    欧拉角的单位是弧度。
    """
    # 初始化输出数组
    euler_angles = np.zeros((quaternions.shape[0], 3))

    for i, (x, y, z, w) in enumerate(quaternions):
        # roll (绕x轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (绕y轴旋转)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 处理90度角的情况
        else:
            pitch = np.arcsin(sinp)

        # yaw (绕z轴旋转)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # 将结果存储到输出数组中
        euler_angles[i] = [roll, pitch, yaw]

    return euler_angles

def compute_velocity(position, dt):
    velocity = (position[1:-1,:] - position[0:-2,:])/dt
    return velocity

# raisim
raisim_obs_path = 'BJ_Raisim/net/HSW/data/actor_state.csv'
# raisim_obs_path = 'BJ_Raisim/net/HSW/data/actor_state_TsingHua.csv'
raisim_torque_path = 'BJ_Raisim/net/HSW/data/torques.csv'
raisim_base_euler_path = 'BJ_Raisim/net/HSW/data/base_euler.csv'

# isaac gym
pose = 'swing'
isaacgym_obs_path = 'BJ_Raisim/net/HSW/data/'+pose+'_obs.csv'
isaacgym_torque_path = 'BJ_Raisim/net/HSW/data/'+pose+'_torque.csv'
isaacgym_base_euler_path = 'BJ_Raisim/net/HSW/data/'+pose+'_base_euler.csv'

# real
real_obs_path = 'deploy/data/robot_data.csv'

# plan
swing_traj_path = '/home/pcpc/robot_dance/opti_traj/output_panda_fixed_gripper/panda_swing.txt'

plot_raisim = False
plot_isaacgym = False
plot_plan = False
plot_real = True

start = 0
end = 1200

default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]

raisim_obs = np.loadtxt(raisim_obs_path, delimiter=",")
isaacgym_obs = np.loadtxt(isaacgym_obs_path, delimiter=",")

isaacgym_torques = np.loadtxt(isaacgym_torque_path, delimiter=",")
raisim_torques = np.loadtxt(raisim_torque_path, delimiter=",")

isaacgym_base_euler = np.loadtxt(isaacgym_base_euler_path, delimiter=',')
raism_base_euler = np.loadtxt(raisim_base_euler_path, delimiter=',')
raisim_base_euler_z_o = raism_base_euler[1000, 2]
raism_base_euler[:, 2] -=  raisim_base_euler_z_o

swing_traj = np.loadtxt(swing_traj_path, delimiter=',')
swing_base_quat = swing_traj[:,3:7]
swing_base_euler = np.zeros((6000,3))
swing_base_euler[:100,:] = quaternion_to_euler(swing_base_quat[:100,:])

# real data
real_data = np.loadtxt(real_obs_path, delimiter=',')
real_obs = real_data[:, :42]
real_command = real_data[:, 42:]


for i in range(99):
    swing_base_euler[40+60*i:40+60*(i+1),:] = swing_base_euler[40:100,:]
swing_base_euler = np.roll(swing_base_euler, shift=30)


isaacgym_torques = isaacgym_torques[start:end, :]
raisim_torques = raisim_torques[start:end, :]
dt = 0.02
# raisim
raisim_ang_vel = raisim_obs[start:end, 0:3]
raisim_projected_gravity = raisim_obs[start:end, 3: 6]
raisim_dof_pos = (raisim_obs[start:end, 6:18] + default_dof_pos)
raisim_dof_v = compute_velocity(raisim_dof_pos,dt)
raisim_actions = np.clip(raisim_obs[start:end, 30:42],-10,10)*0.25 + default_dof_pos
# raisim_actions = raisim_obs[start:end, 24:42]*0.25 + default_dof_pos

# isaac gym
isaacgym_ang_vel = isaacgym_obs[start:end, 0:3]
isaacgym_projected_gravity = isaacgym_obs[start:end, 3: 6]
isaacgym_dof_pos = isaacgym_obs[start:end, 6:18] + default_dof_pos
isaacgym_actions = np.clip(isaacgym_obs[start:end, 30:42],-10,10)*0.25 + default_dof_pos

# real
real_ang_vel = real_obs[start:end, :3]
real_projected_gravity = real_obs[start:end, 3: 6]
real_dof_pos = (real_obs[start:end, 6:18] + default_dof_pos)
real_dof_v = compute_velocity(real_dof_pos,dt)
real_actions = np.clip(real_obs[start:end, 30:42],-10,10)*0.25 + default_dof_pos


nb_rows = 3
nb_cols = 3
plt.rcParams['font.size'] = 20
fig, axs = plt.subplots(1, 2)

# 机身角速度
a = axs[0,]
a.grid(True)
if plot_raisim:
    a.plot(raisim_ang_vel[:, 0], label='raisim_ang_vel_x', c='r')
    a.plot(raisim_ang_vel[:, 1], label='raisim_ang_vel_y', c='g')
    a.plot(raisim_ang_vel[:, 2], label='raisim_ang_vel_z', c='b')
if plot_isaacgym:
    a.plot(isaacgym_ang_vel[:, 0], label='isaacgym_ang_vel_x', linestyle='--', c='r')
    a.plot(isaacgym_ang_vel[:, 1], label='isaacgym_ang_vel_y', linestyle='--', c='g')
    a.plot(isaacgym_ang_vel[:, 2], label='isaacgym_ang_vel_z', linestyle='--', c='b')
if plot_real:
    a.plot(real_ang_vel[:, 0], label='real_ang_vel_x', c='r')
    a.plot(real_ang_vel[:, 1], label='real_ang_vel_y', c='g')
    a.plot(real_ang_vel[:, 2], label='real_ang_vel_z', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='base ang vel')
a.legend()
# 投影重力
a = axs[1,]
a.grid(True)
if plot_raisim:
    a.plot(raisim_projected_gravity[:, 0], label='raisim_g_x', c='r')
    a.plot(raisim_projected_gravity[:, 1], label='raisim_g_y', c='g')
    a.plot(raisim_projected_gravity[:, 2], label='raisim_g_z', c='b')
if plot_isaacgym:
    a.plot(isaacgym_projected_gravity[:, 0], label='isaacgym_g_x', linestyle='--', c='r')
    a.plot(isaacgym_projected_gravity[:, 1], label='isaacgym_g_y', linestyle='--', c='g')
    a.plot(isaacgym_projected_gravity[:, 2], label='isaacgym_g_z', linestyle='--', c='b')
if plot_real:
    a.plot(real_projected_gravity[:, 0], label='real_g_x', c='r')
    a.plot(real_projected_gravity[:, 1], label='real_g_y', c='g')
    a.plot(real_projected_gravity[:, 2], label='real_g_z', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='projected gravity')
a.legend()
# 关节角度
fig2, axs2 = plt.subplots(2, 2)
a = axs2[0, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raisim_dof_pos[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raisim_dof_pos[:, 2], label='raisim_LF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 0], label='isaacgym_LF_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 1], label='isaacgym_LF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 2], label='isaacgym_LF_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_pos[:, 0], label='real_LF_hip', c='r')
    a.plot(real_dof_pos[:, 1], label='real_LF_thigh', c='g')
    a.plot(real_dof_pos[:, 2], label='real_LF_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs2[0, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 3], label='raisim_RF_hip', c='r')
    a.plot(raisim_dof_pos[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(raisim_dof_pos[:, 5], label='raisim_RF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 3], label='isaacgym_RF_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 4], label='isaacgym_RF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 5], label='isaacgym_RF_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_pos[:, 3], label='real_RF_hip', c='r')
    a.plot(real_dof_pos[:, 4], label='real_RF_thigh', c='g')
    a.plot(real_dof_pos[:, 5], label='real_RF_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs2[1, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 6], label='raisim_LH_hip', c='r')
    a.plot(raisim_dof_pos[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(raisim_dof_pos[:, 8], label='raisim_LH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 6], label='isaacgym_LH_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 7], label='isaacgym_LH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 8], label='isaacgym_LH_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_pos[:, 6], label='real_LH_hip', c='r')
    a.plot(real_dof_pos[:, 7], label='real_LH_thigh', c='g')
    a.plot(real_dof_pos[:, 8], label='real_LH_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs2[1, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 9], label='raisim_RH_hip', c='r')
    a.plot(raisim_dof_pos[:, 10], label='raisim_RH_thigh', c='g')
    a.plot(raisim_dof_pos[:, 11], label='raisim_RH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 9], label='isaacgym_RH_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 10], label='isaacgym_RH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 11], label='isaacgym_RH_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_pos[:, 9], label='real_RH_hip', c='r')
    a.plot(real_dof_pos[:, 10], label='real_RH_thigh', c='g')
    a.plot(real_dof_pos[:, 11], label='real_RH_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

# 关节角速度
fig8, axs8 = plt.subplots(2, 2)
a = axs8[0, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_v[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raisim_dof_v[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raisim_dof_v[:, 2], label='raisim_LF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 0], label='isaacgym_LF_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 1], label='isaacgym_LF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 2], label='isaacgym_LF_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_v[:, 0], label='real_LF_hip', c='r')
    a.plot(real_dof_v[:, 1], label='real_LF_thigh', c='g')
    a.plot(real_dof_v[:, 2], label='real_LF_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof velocity')
# a.legend()

a = axs8[0, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_v[:, 3], label='raisim_RF_hip', c='r')
    a.plot(raisim_dof_v[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(raisim_dof_v[:, 5], label='raisim_RF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 3], label='isaacgym_RF_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 4], label='isaacgym_RF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 5], label='isaacgym_RF_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_v[:, 3], label='real_RF_hip', c='r')
    a.plot(real_dof_v[:, 4], label='real_RF_thigh', c='g')
    a.plot(real_dof_v[:, 5], label='real_RF_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof velocity')
# a.legend()

a = axs8[1, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_v[:, 6], label='raisim_LH_hip', c='r')
    a.plot(raisim_dof_v[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(raisim_dof_v[:, 8], label='raisim_LH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 6], label='isaacgym_LH_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 7], label='isaacgym_LH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 8], label='isaacgym_LH_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_v[:, 6], label='real_LH_hip', c='r')
    a.plot(real_dof_v[:, 7], label='real_LH_thigh', c='g')
    a.plot(real_dof_v[:, 8], label='real_LH_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof velocity')
# a.legend()

a = axs8[1, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_v[:, 9], label='raisim_RH_hip', c='r')
    a.plot(raisim_dof_v[:, 10], label='raisim_RH_thigh', c='g')
    a.plot(raisim_dof_v[:, 11], label='raisim_RH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_dof_pos[:, 9], label='isaacgym_RH_hip', linestyle='--', c='r')
    a.plot(isaacgym_dof_pos[:, 10], label='isaacgym_RH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_dof_pos[:, 11], label='isaacgym_RH_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_dof_v[:, 9], label='real_RH_hip', c='r')
    a.plot(real_dof_v[:, 10], label='real_RH_thigh', c='g')
    a.plot(real_dof_v[:, 11], label='real_RH_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof velocity')
# a.legend()

# last_action
fig3, axs3 = plt.subplots(2, 2)
a = axs3[0, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_actions[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raisim_actions[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raisim_actions[:, 2], label='raisim_LF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 0], label='isaacgym_LF_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 1], label='isaacgym_LF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 2], label='isaacgym_LF_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_actions[:, 0], label='raisim_LF_hip', c='r')
    a.plot(real_actions[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(real_actions[:, 2], label='raisim_LF_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')

a = axs3[0, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_actions[:, 3], label='raisim_RF_hip', c='r')
    a.plot(raisim_actions[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(raisim_actions[:, 5], label='raisim_RF_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 3], label='isaacgym_RF_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 4], label='isaacgym_RF_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 5], label='isaacgym_RF_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_actions[:, 3], label='raisim_RF_hip', c='r')
    a.plot(real_actions[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(real_actions[:, 5], label='raisim_RF_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')

a = axs3[1, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_actions[:, 6], label='raisim_LH_hip', c='r')
    a.plot(raisim_actions[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(raisim_actions[:, 8], label='raisim_LH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 6], label='isaacgym_LH_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 7], label='isaacgym_LH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 8], label='isaacgym_LH_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_actions[:, 6], label='raisim_LH_hip', c='r')
    a.plot(real_actions[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(real_actions[:, 8], label='raisim_LH_calf', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='actions')

a = axs3[1, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_actions[:, 9], label='raisim_RH_hip', c='r')
    a.plot(raisim_actions[:, 10], label='raisim_RH_thigh', c='g')
    a.plot(raisim_actions[:, 11], label='raisim_RH_calf', c='b')
if plot_isaacgym:
    a.plot(isaacgym_actions[:, 9], label='isaacgym_RH_hip', linestyle='--', c='r')
    a.plot(isaacgym_actions[:, 10], label='isaacgym_RH_thigh', linestyle='--', c='g')
    a.plot(isaacgym_actions[:, 11], label='isaacgym_RH_calf', linestyle='--', c='b')
if plot_real:
    a.plot(real_actions[:, 9], label='raisim_RH_hip', c='r')
    a.plot(real_actions[:, 10], label='raisim_RH_thigh', c='g')
    a.plot(real_actions[:, 11], label='raisim_RH_calf', c='b')
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
a.grid(True)
if plot_raisim:
    a.plot(raism_base_euler[:, 0], label='raisim_base_x', c='r')
    a.plot(raism_base_euler[:, 1], label='raisim_base_y', c='g')
    a.plot(raism_base_euler[:, 2], label='raisim_base_z', c='b')
if plot_isaacgym:
    a.plot(isaacgym_base_euler[:, 0], label='isaacgym_base_x', linestyle='--', c='r')
    a.plot(isaacgym_base_euler[:, 1], label='isaacgym_base_y', linestyle='--', c='g')
    a.plot(isaacgym_base_euler[:, 2], label='isaacgym_base_z', linestyle='--', c='b')
if plot_plan:
    # a.plot(isaacgym_base_euler[:, 0], label='plan_base_x', linestyle='--', c='r')
    # a.plot(isaacgym_base_euler[:, 1], label='plan_base_y', linestyle='--', c='g')
    a.plot(swing_base_euler[:, 2], label='plan_base_z', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='base euler')
a.legend()

# plot base_euler
fig7, axs7 = plt.subplots()
a = axs7
a.grid(True)
if plot_plan:
    # a.plot(isaacgym_base_euler[:, 0], label='isaacgym_base_x', linestyle='--', c='r')
    # a.plot(isaacgym_base_euler[:, 1], label='isaacgym_base_y', linestyle='--', c='g')
    a.plot(swing_base_euler[:, 2], label='plan_base_z', linestyle='--', c='b')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='base euler')

# raisim关节角度 vs 期望关节角度
fig6, axs6 = plt.subplots(2, 2)
a = axs6[0, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 0], label='raisim_LF_hip', c='r')
    a.plot(raisim_dof_pos[:, 1], label='raisim_LF_thigh', c='g')
    a.plot(raisim_dof_pos[:, 2], label='raisim_LF_calf', c='b')

    a.plot(raisim_actions[:, 0], label='raisim_LF_hip', c='r', linestyle='--')
    a.plot(raisim_actions[:, 1], label='raisim_LF_thigh', c='g', linestyle='--')
    a.plot(raisim_actions[:, 2], label='raisim_LF_calf', c='b', linestyle='--')
if plot_real:
    a.plot(real_dof_pos[:, 0], label='real_LF_hip', c='r')
    a.plot(real_dof_pos[:, 1], label='real_LF_thigh', c='g')
    a.plot(real_dof_pos[:, 2], label='real_LF_calf', c='b')

    a.plot(real_command[:, 0], label='real_LF_hip', c='r', linestyle='--')
    a.plot(real_command[:, 1], label='real_LF_thigh', c='g', linestyle='--')
    a.plot(real_command[:, 2], label='real_LF_calf', c='b', linestyle='--')

plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs6[0, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 3], label='raisim_RF_hip', c='r')
    a.plot(raisim_dof_pos[:, 4], label='raisim_RF_thigh', c='g')
    a.plot(raisim_dof_pos[:, 5], label='raisim_RF_calf', c='b')

    a.plot(raisim_actions[:, 3], label='raisim_RF_hip', c='r', linestyle='--')
    a.plot(raisim_actions[:, 4], label='raisim_RF_thigh', c='g', linestyle='--')
    a.plot(raisim_actions[:, 5], label='raisim_RF_calf', c='b', linestyle='--')
if plot_real:
    a.plot(real_dof_pos[:, 3], label='real_RF_hip', c='r')
    a.plot(real_dof_pos[:, 4], label='real_RF_thigh', c='g')
    a.plot(real_dof_pos[:, 5], label='real_RF_calf', c='b')

    a.plot(real_command[:, 3], label='real_RF_hip', c='r', linestyle='--')
    a.plot(real_command[:, 4], label='real_RF_thigh', c='g', linestyle='--')
    a.plot(real_command[:, 5], label='real_RF_calf', c='b', linestyle='--')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs6[1, 0]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 6], label='raisim_LH_hip', c='r')
    a.plot(raisim_dof_pos[:, 7], label='raisim_LH_thigh', c='g')
    a.plot(raisim_dof_pos[:, 8], label='raisim_LH_calf', c='b')

    a.plot(raisim_actions[:, 6], label='raisim_LH_hip', c='r', linestyle='--')
    a.plot(raisim_actions[:, 7], label='raisim_LH_thigh', c='g', linestyle='--')
    a.plot(raisim_actions[:, 8], label='raisim_LH_calf', c='b', linestyle='--')
if plot_real:
    a.plot(real_dof_pos[:, 6], label='real_LH_hip', c='r')
    a.plot(real_dof_pos[:, 7], label='real_LH_thigh', c='g')
    a.plot(real_dof_pos[:, 8], label='real_LH_calf', c='b')

    a.plot(real_command[:, 6], label='real_LH_hip', c='r', linestyle='--')
    a.plot(real_command[:, 7], label='real_LH_thigh', c='g', linestyle='--')
    a.plot(real_command[:, 8], label='real_LH_calf', c='b', linestyle='--')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()

a = axs6[1, 1]
a.grid(True)
if plot_raisim:
    a.plot(raisim_dof_pos[:, 9], label='raisim_LH_hip', c='r')
    a.plot(raisim_dof_pos[:, 10], label='raisim_LH_thigh', c='g')
    a.plot(raisim_dof_pos[:, 11], label='raisim_LH_calf', c='b')

    a.plot(raisim_actions[:, 9], label='raisim_RH_hip', c='r', linestyle='--')
    a.plot(raisim_actions[:, 10], label='raisim_RH_thigh', c='g', linestyle='--')
    a.plot(raisim_actions[:, 11], label='raisim_RH_calf', c='b', linestyle='--')
if plot_real:
    a.plot(real_dof_pos[:, 9], label='real_RH_hip', c='r')
    a.plot(real_dof_pos[:, 10], label='real_RH_thigh', c='g')
    a.plot(real_dof_pos[:, 11], label='real_RH_calf', c='b')

    a.plot(real_command[:, 9], label='real_RH_hip', c='r', linestyle='--')
    a.plot(real_command[:, 10], label='real_RH_thigh', c='g', linestyle='--')
    a.plot(real_command[:, 11], label='real_RH_calf', c='b', linestyle='--')
plt.rcParams['xtick.labelsize'] = 20
a.set(title='dof pos')
# a.legend()


plt.show()
