import numpy as np
from pybullet_utils.transformations import quaternion_slerp

# 规划转身跳跃动作的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 机身右转30°，左转60°，右转60°，左转30°
num_row = 80
num_col = 49
fps = 50
g = 9.81

turn_and_jump_ref = np.ones((num_row, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))



def h_1(t):
    x = 0.2  # 对称轴
    h = g / 2 * x ** 2
    return h - g / 2 * (t - x) ** 2

class ROBOT:
    def __init__(self,leg_length):
        self.RF_leg = []


# 质心轨迹
for i in range(20):
    root_pos[i, 2] = h_1(i / 50) + 0.3
for i in range(3):
    root_pos[20 * (i + 1):20 * (i + 2), :] = root_pos[:20, :]

# 姿态
q1 = [0, 0, 0, 1]  # [x,y,z,w]
q2 = [0, 0, np.sin(-np.pi / 12), np.cos(-np.pi / 12)]
q3 = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
interval = 20
start = 0
end = start + interval

for i in range(end):
    frac = i / end
    root_rot[i,:] = quaternion_slerp(q1, q2, frac)

start = end
end = start+interval
for i in range(start,end):
    frac = (i-start)/(end-start)
    root_rot[i, :] = quaternion_slerp(q2, q3, frac)

start = end
end = start+interval
for i in range(start,end):
    frac = (i-start)/(end-start)
    root_rot[i, :] = quaternion_slerp(q3, q2, frac)

start = end
end = start+interval
for i in range(start,end):
    frac = (i-start)/(end-start)
    root_rot[i, :] = quaternion_slerp(q2, q1, frac)

# 读取文件
motion_files = 'output/keep_the_beat/keep_the_beat_ref_simp.txt'
motion_data = np.loadtxt(motion_files, delimiter=',')
toe_pos = motion_data[0,13:25]

# 组合轨迹
turn_and_jump_ref[:,:3] = root_pos
turn_and_jump_ref[:,3:7] = root_rot
turn_and_jump_ref[:,13:25] = toe_pos

# 导出txt
outfile = 'output/turn_and_jump.txt'
np.savetxt(outfile,turn_and_jump_ref,delimiter=',')

