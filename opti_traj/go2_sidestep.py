import numpy as np
import utils
import casadi as ca
import CPG
import json

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告：matplotlib未安装，将跳过可视化部分")

# 使用CPG模型设计侧向pace步态，机器人侧向迈步并回到原位
# pace步态：同侧腿同时抬起，相位差为[0, π, 0, π]

num_row = 200  # 增加帧数以容纳完整的侧向迈步动作
num_col = 49
fps = 50

sidestep_ref = np.ones((num_row-1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row-1, 3))
root_ang_vel = np.zeros((num_row-1, 3))
root_rot_dot = np.zeros((num_row-1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row-1, 12))
hopf_signal = np.zeros((num_row, 8))

go2 = utils.QuadrupedRobot()
initPos = np.array([-0.5, 0.5, 0.5, -0.5, 0, 0, 0, 0])
gait = 'pace'
cpg = CPG.cpgBuilder(initPos, gait=gait)

# 欧拉法获取振荡信号
t = np.linspace(0, num_row/fps, num_row)
temp = cpg.initPos.reshape(8, -1)

for i in range(num_row):
    v = cpg.hopf_osci(temp)
    temp += v/fps
    hopf_signal[i] = temp.flatten()

# 相位
phase = np.arctan2(hopf_signal[:, 4:8], hopf_signal[:, 0:4])

# 设计侧向迈步轨迹
# 分为三个阶段：
# 1. 前半程：向右侧迈步 (0 -> num_row//2)
# 2. 后半程：回到原位 (num_row//2 -> num_row)

vx = 0.0  # x方向不移动
vy_max = 0.4  # 最大侧向速度
az = 0.08  # 抬腿高度

# 计算每个时刻的侧向速度和位移
for i in range(num_row):
    progress = i / (num_row - 1)
    
    if progress <= 0.5:
        # 前半程：向右侧迈步
        # 使用正弦函数实现平滑加速和减速
        vy_current = vy_max * np.sin(2 * np.pi * progress)
        root_lin_vel[min(i, num_row-2), 1] = vy_current
    else:
        # 后半程：回到原位
        # 使用负正弦函数回到原位
        vy_current = -vy_max * np.sin(2 * np.pi * (progress - 0.5))
        root_lin_vel[min(i, num_row-2), 1] = vy_current

# 计算质心的y方向位移
y_pos = 0.0
for i in range(num_row):
    if i > 0:
        y_pos += root_lin_vel[i-1, 1] / fps
    root_pos[i, 1] = y_pos

# 足端轨迹计算
ay_amplitude = vy_max * cpg.T * cpg.beta  # 侧向足端轨迹幅度

for i in range(4):
    for j in range(num_row):
        if phase[j, i] < 0:
            p = -phase[j, i] / np.pi
            toe_pos[j, 3*i+2] = CPG.endEffectorPos_z(az, p)
        else:
            p = phase[j, i] / np.pi
            toe_pos[j, 3*i+2] = 0

        # x方向保持不变
        toe_pos[j, 3*i] = 0
        
        # y方向的足端轨迹：根据步态阶段调整
        progress = j / (num_row - 1)
        if progress <= 0.5:
            # 前半程：向右侧迈步
            toe_pos[j, 3*i+1] = CPG.endEffectorPos_xy(ay_amplitude, p)
        else:
            # 后半程：回到原位，足端轨迹反向
            toe_pos[j, 3*i+1] = -CPG.endEffectorPos_xy(ay_amplitude, p)

# 足端相对质心的坐标
toe_pos += go2.toe_pos_init
q = ca.SX.sym('q', 3, 1)

# 逆运动学求解关节角度
for j in range(4):
    for i in range(num_row):
        # toe_pos是质心系下的足端轨迹
        pos = go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        cost = 500 * ca.dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), 
                           (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = ca.nlpsol('S', 'ipopt', nlp, {'print_time': 0, 'ipopt': {'print_level': 0}})
        r = S(x0=[0.1, 0.8, -1.5], lbx=go2.lb[3*j:3*j+3], ubx=go2.ub[3*j:3*j+3])
        q_opt = r['x']
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i+1, :] - dof_pos[i, :]) * fps

# 质心位置设置
root_pos[:, 0] = 0  # x方向不移动
root_pos[:, 2] = 0.28  # 保持固定高度

# 机身方向（四元数）
root_rot[:, 3] = 1  # w = 1, 表示无旋转

# 机身角速度默认为0
root_ang_vel[:, :] = 0

# 组合轨迹
sidestep_ref[:, :3] = root_pos[:num_row-1, :]
sidestep_ref[:, 3:7] = root_rot[:num_row-1, :]
sidestep_ref[:, 7:10] = root_lin_vel
sidestep_ref[:, 10:13] = root_ang_vel
sidestep_ref[:, 13:25] = toe_pos[:num_row-1, :]
sidestep_ref[:, 25:37] = dof_pos[:num_row-1, :]
sidestep_ref[:, 37:49] = dof_vel

# 导出txt文件
outfile = 'output/sidestep.txt'
np.savetxt(outfile, sidestep_ref, delimiter=',')

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': sidestep_ref.tolist()
}

with open('output_json/sidestep.json', 'w') as f:
    json.dump(json_data, f, indent=4)

print(f"侧向pace步态轨迹已生成并保存到:")
print(f"- {outfile}")
print(f"- output_json/sidestep.json")
print(f"总帧数: {num_row-1}, 持续时间: {(num_row-1)/fps:.2f}秒")
print(f"最大侧向位移: {np.max(np.abs(root_pos[:, 1])):.3f}米")

# 可视化轨迹（如果matplotlib可用）
if MATPLOTLIB_AVAILABLE:
    plt.figure(figsize=(12, 8))

    # 子图1：质心y方向位移
    plt.subplot(2, 2, 1)
    plt.plot(t, root_pos[:, 1], 'b-', linewidth=2, label='质心Y位移')
    plt.xlabel('时间 (s)')
    plt.ylabel('Y位移 (m)')
    plt.title('质心侧向位移轨迹')
    plt.grid(True)
    plt.legend()

    # 子图2：质心y方向速度
    plt.subplot(2, 2, 2)
    plt.plot(t[:-1], root_lin_vel[:, 1], 'r-', linewidth=2, label='质心Y速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('Y速度 (m/s)')
    plt.title('质心侧向速度')
    plt.grid(True)
    plt.legend()

    # 子图3：相位信号
    plt.subplot(2, 2, 3)
    plt.plot(t, phase[:, 0], linewidth=2, label='FR腿相位')
    plt.plot(t, phase[:, 1], linewidth=2, label='FL腿相位')
    plt.plot(t, phase[:, 2], linewidth=2, label='HR腿相位')
    plt.plot(t, phase[:, 3], linewidth=2, label='HL腿相位')
    plt.xlabel('时间 (s)')
    plt.ylabel('相位 (rad)')
    plt.title('Pace步态相位信号')
    plt.grid(True)
    plt.legend()

    # 子图4：足端轨迹示例（FR腿）
    plt.subplot(2, 2, 4)
    plt.plot(toe_pos[:, 1], toe_pos[:, 2], 'g-', linewidth=2, label='FR腿足端轨迹')
    plt.xlabel('Y位置 (m)')
    plt.ylabel('Z位置 (m)')
    plt.title('前右腿足端轨迹')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()
else:
    print("跳过可视化，matplotlib未安装")