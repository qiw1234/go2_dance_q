import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *

# 规划挥手的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]

# 挥手动作：右前腿抬起来，挥舞右前腿，轨迹轮廓在y-z平面来看类似一片银杏叶
# 包含两段轨迹，一是抬腿到指定位置，不在一个平面内，需要通过两个平面图形进行组合，二是在yz平面内挥舞
# 由于机器狗的手臂关节的结构，横着挥手感觉不太好看，所以我想着做一个招财猫的动作
# 具体来说就是身体仰起来30°，然后右手做招财猫的动作

# 实例化panda7
# panda7的关节上下限
panda_lb = [-0.87, -1.78, -2.53, -0.69, -1.78, -2.53, -0.87, -1.3, -2.53, -0.69, -1.3, -2.53]
panda_ub = [0.69, 3.4, -0.45, 0.87, 3.4, -0.45, 0.69, 4, -0.45, 0.87, 4, -0.45]
panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.287854, -0.481828, -0.349867,
                      -0.287854, -0.481828, -0.349867, 0.287854, -0.481828]
# 左前腿向内收，这样能三条腿能站稳
# panda_toe_pos_init = [0.300133, -0.287854, -0.481828, 0.300133, 0.0353867, -0.506912, -0.349867,
#                       -0.287854, -0.481828, -0.349867, 0.287854, -0.481828] # 关节角度-0.4
panda7 = utils.QuadrupedRobot(l=0.65, w=0.225, l1=0.126375, l2=0.34, l3=0.34,
                              lb=panda_lb, ub=panda_ub, toe_pos_init=panda_toe_pos_init)
num_row = 80
num_col = 72
fps = 50

ref = np.ones((num_row - 1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row - 1, 3))
root_ang_vel = np.zeros((num_row - 1, 3))
root_rot_dot = np.zeros((num_row - 1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row - 1, 12))
arm_pos = np.zeros((num_row, 3))
arm_rot = np.zeros((num_row, 4))
arm_dof_pos = np.zeros((num_row, 8))
arm_dof_vel = np.zeros((num_row-1, 8))

def z_1(y, y0, z0):
    z = z0 - z0 * (y + y0) ** 2 / (y0 ** 2)
    return z

# 质心轨迹
root_pos[:, 2] = 0.55
# 质心线速度 默认为0

# 姿态
q0 = [0, 0, 0, 1]
q1 = [0, np.sin(-np.pi / 24), 0, np.cos(-np.pi / 24)]
end = 10
for i in range(end):
    frac = i / (end - 1)
    root_rot[i, :] = quaternion_slerp(q0, q1, frac)
start = end
end = 70
root_rot[start: end, :] = root_rot[start - 1, :]
start = end
end = num_row
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q1, q0, frac)


# 四元数的导数
for i in range(num_row-1):
    root_rot_dot[i,:] = (root_rot[i+1,:] - root_rot[i,:]) * fps
# 质心角速度
for i in range(num_row-1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i,:])@ root_rot_dot[i,:]

# 足端轨迹
# 除右前腿，其他的都是世界系的位置，都保持不动
# 这是默认的足端位置，坐标系是固定在root处的世界系
# 右前腿的动作通过关节角度规划
toe_pos[:] = panda7.toe_pos_init
q_FR_0 = [-0.1, 0.8, -1.5]  # 初始位置
q_FR_1 = [-0.1, -0.8, -1.5]  # 抬手中间位置
q_FR_2 = [-0.1, -0.8, -1]  # 抬手下方
q_FR_3 = [-0.1, -0.8, -2]  # 抬手上方
q_FL_0 = [0.1, 0.8, -1.5]
q_FL_1 = [-0.4, 0.8, -1.5]
# 右前腿关节角度
dof_pos[:10, :3] = q_FR_0
dof_pos[10:20, :3] = np.linspace(q_FR_0, q_FR_1, 10)
dof_pos[20:30, :3] = np.linspace(q_FR_1, q_FR_2, 10)
dof_pos[30:40, :3] = np.linspace(q_FR_2, q_FR_3, 10)
dof_pos[40:50, :3] = np.linspace(q_FR_3, q_FR_2, 10)
dof_pos[50:60, :3] = np.linspace(q_FR_2, q_FR_3, 10)
dof_pos[60:70, :3] = np.linspace(q_FR_3, q_FR_0, 10)
dof_pos[70:80, :3] = q_FR_0
# 左前腿关节角度
dof_pos[:, 3:6] = q_FL_1
dof_pos[:10, 3:6] = np.linspace(q_FL_0, q_FL_1, 10)
dof_pos[70:80, 3:6] = np.linspace(q_FL_1, q_FL_0, 10)

# 计算足端位置在质心坐标系的坐标
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = np.transpose(casadi.DM(panda7.transrpy(dof_pos[i, :3], 0, [0, 0, 0], [0, 0, 0]) @ panda7.toe).full()[:3])
    toe_pos[i, 3:6] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ toe_pos[i, 3:6]
    toe_pos[i, 6:9] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ toe_pos[i, 6:9]
    toe_pos[i, 9:12] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ toe_pos[i, 9:12]

# go2的关节上下限
q = SX.sym('q', 3, 1)
for j in range(1, 4):
    for i in range(num_row):
        # 这里的toe_pos是世界系足端轨迹，需要考虑质心姿态，因此左乘一个质心姿态
        pos = (panda7.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ panda7.toe)[:3]
        cost = 500 * casadi.dot((toe_pos[i, 3 * j:3 * j + 3] - pos[:3]), (toe_pos[i, 3 * j:3 * j + 3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = casadi.nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=panda7.lb[3 * j:3 * j + 3], ubx=panda7.ub[3 * j:3 * j + 3])
        q_opt = r['x']
        dof_pos[i, 3 * j:3 * j + 3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i + 1, :] - dof_pos[i, :]) * fps

# arm fk
robot_arm_rot, robot_arm_pos=utils.arm_fk([0, 0, 0, 0, 0, 0])
# 机械臂末端在机身坐标系下的位置
arm_pos[:] = robot_arm_pos
# 机械臂末端在世界系下的姿态
for i in range(num_row):
    arm_rot[i, :] = utils.rotm2quaternion(utils.quaternion2rotm(root_rot[i,:]) @ robot_arm_rot)



# 组合轨迹
# 最终输出的末端位置是在世界系中，末端相对质心的位置
ref[:, :3] = root_pos[:num_row - 1, :]
ref[:, 3:7] = root_rot[:num_row - 1, :]
ref[:, 7:10] = root_lin_vel
ref[:, 10:13] = root_ang_vel
ref[:, 13:25] = toe_pos[:num_row - 1, :]
ref[:, 25:37] = dof_pos[:num_row - 1, :]
ref[:, 37:49] = dof_vel
ref[:, 49:52] = arm_pos[:num_row - 1, :]
ref[:, 52:56] = arm_rot[:num_row - 1, :]
ref[:, 56:64] = arm_dof_pos[:num_row - 1, :]
ref[:, 64:72] = arm_dof_vel

# # 导出完整轨迹
outfile = 'output_panda/panda_wave.txt'
np.savetxt(outfile, ref, delimiter=',')

# # 导出fixed arm轨迹
outfile = 'output_panda_fixed_arm/panda_wave.txt'
np.savetxt(outfile, ref[:, :49], delimiter=',')

# 导出 fixed gripper轨迹
outfile = 'output_panda_fixed_gripper/panda_wave.txt'
out = np.hstack((ref[:, :56], ref[:, 56:62], ref[:, 64:70]))
np.savetxt(outfile, out, delimiter=',')
