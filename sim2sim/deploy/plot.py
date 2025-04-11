import numpy as np
import matplotlib.pyplot as plt

# 读取仿真数据和实物数据
sim_data = np.loadtxt('/home/lw/PycharmProjects/panda7_robot_dance_project/go2/BJ_Raisim/net/HSW/data/actor_state_jump20.csv', delimiter=',')
real_data = np.loadtxt('data/actorState_jump20ms.csv', delimiter=',')
# 选取同样区间的数据（比如第100到500帧）
sim_data = sim_data[200:500]
real_data = real_data[200:500]

# 分解数据（仿真）
sim_joint_angles = sim_data[:, 6:18]
sim_joint_velocities = sim_data[:, 18:30]
sim_action_values = sim_data[:, 30:42]

# 分解数据（实物）
real_joint_angles = real_data[:, 6:18]
real_joint_velocities = real_data[:, 18:30]
real_action_values = real_data[:, 30:42]

# ------- 绘制关节角度对比 --------
plt.figure(figsize=(12, 6))
for i in range(sim_joint_angles.shape[1]):
    plt.plot(sim_joint_angles[:, i], label=f'Sim Joint {i+1}', linestyle='--')
    plt.plot(real_joint_angles[:, i], label=f'Real Joint {i+1}', linestyle='-')
plt.title('Joint Angles Comparison')
plt.xlabel('Time Step')
plt.ylabel('Angle (radians)')
plt.legend(ncol=4, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
#
# # ------- 绘制关节角速度对比 --------
# plt.figure(figsize=(12, 6))
# for i in range(sim_joint_velocities.shape[1]):
#     plt.plot(sim_joint_velocities[:, i], label=f'Sim Vel {i+1}', linestyle='--')
#     plt.plot(real_joint_velocities[:, i], label=f'Real Vel {i+1}', linestyle='-')
# plt.title('Joint Velocities Comparison')
# plt.xlabel('Time Step')
# plt.ylabel('Velocity (radians/s)')
# plt.legend(ncol=4, fontsize=8)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # ------- 绘制动作值对比 --------
# plt.figure(figsize=(12, 6))
# for i in range(sim_action_values.shape[1]):
#     plt.plot(sim_action_values[:, i], label=f'Sim Action {i+1}', linestyle='--')
#     plt.plot(real_action_values[:, i], label=f'Real Action {i+1}', linestyle='-')
# plt.title('Action Values Comparison')
# plt.xlabel('Time Step')
# plt.ylabel('Action Value')
# plt.legend(ncol=4, fontsize=8)
# plt.grid(True)
# plt.tight_layout()

# ------- 绘制机身高度（Z轴）对比 --------
sim_body_height = sim_data[:, 2]  # 机身高度，假设第2列是Z
real_body_height = real_data[:, 2]

plt.figure(figsize=(10, 5))
plt.plot(sim_body_height, label='Sim Body Height', linestyle='--')
plt.plot(real_body_height, label='Real Body Height', linestyle='-')
plt.title('Body Height (Z) Comparison')
plt.xlabel('Time Step')
plt.ylabel('Height (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()