import matplotlib.pyplot as plt
import numpy as np

# LF RF LH RH
default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5])
# 更换腿部顺序
reindex_feet = [1, 0, 3, 2]
# task = '0~100msstand'
# task = 'trot'
task = None
if task is not None:
    real_obs_path = 'deploy/data/real/' + task + '/actorState.csv'
    real_cmd_path = 'deploy/data/real/' + task + '/cmd.csv'
    real_time_path = 'deploy/data/real/' + task + '/T.csv'
    raisim_obs_path = 'BJ_Raisim/net/HSW/data/' + task + '/actor_state.csv'
    isaacgym_obs_path = 'BJ_Raisim/net/HSW/data/' + task + '_obs.csv'
else:
    real_obs_path = 'deploy/data/actorState.csv'
    real_cmd_path = 'deploy/data/cmd.csv'
    real_time_path = 'deploy/data/T.csv'
    raisim_obs_path = 'BJ_Raisim/net/HSW/data/actor_state.csv'
# ------------------raisim---------------------------------

raisim_obs = np.loadtxt(raisim_obs_path, delimiter=",")

# ---------------------real------------------------------
# real_obs_path = 'deploy/data/robot_data.csv'
real_data = np.loadtxt(real_obs_path, delimiter=',')
real_obs = real_data[:, :42]
real_dof_pos = real_obs[:, 6:18] + default_dof_pos
real_cmd = np.loadtxt(real_cmd_path, delimiter=',')
real_time = np.loadtxt(real_time_path, delimiter=',')
# isaac gym
task='stand'
isaacgym_obs_path = 'BJ_Raisim/net/HSW/data/' + task + '_obs.csv'
isaacgym_obs = np.loadtxt(isaacgym_obs_path, delimiter=",")
# 网络输入
nb_rows = 7
nb_cols = 6
fig1, axs1 = plt.subplots(nb_rows, nb_cols)
fig1.subplots_adjust(
    wspace=0.3,  # 列间距（水平间距）
    hspace=0.67   # 行间距（垂直间距）
)
start = 950
# end = real_data.shape[0]
end = 1250
for i in range(nb_rows):
    for j in range(nb_cols):
        a = axs1[i,j]
        a.grid(True)
        a.plot(isaacgym_obs[start+185:end+185, nb_cols * i + j], c='b')
        a.plot(raisim_obs[start:end, nb_cols*i+j], c='r')
        # a.plot(real_obs[start:end, nb_cols*i+j], c='g')
        a.set(title=f'OBS_{nb_cols*i+j}')

# # 实物响应速度
# fig2, axs2 = plt.subplots(4, 3)
# fig2.subplots_adjust(hspace=0.35)
# for i in range(4):
#     for j in range(3):
#         a = axs2[i,j]
#         a.grid(True)
#         a.plot(real_time, real_dof_pos[:, 3*i+j]) # 左前右前左后右后
#         a.plot(real_time, real_cmd[:, 3*reindex_feet[i]+j], linestyle='--')
#         a.set(title=f'leg number {3*i+j}')

plt.show()

