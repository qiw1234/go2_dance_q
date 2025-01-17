# -*- coding: utf-8 -*-
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time

def s(x):
    return math.sin(x)


def c(x):
    return math.cos(x)


def t(x):
    return math.tan(x)

reindex_feet1 = [1, 0, 3, 2]

def quat_rotate_inverse(q, v):  # 获取基座z轴在惯性系下的投影矢量，q是IMU的四元数，v是z轴向量 (0,0,-1)
    """
    使用逆四元数旋转向量。

    参数:
        q (torch.Tensor): 四元数，形状为 [batch_size, 4]，格式为 [x, y, z, w]。
        v (torch.Tensor): 要旋转的向量，形状为 [batch_size, 3]。

    返回:
        torch.Tensor: 旋转后的向量，形状为 [batch_size, 3]。
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def rpy2quaternion(roll, pitch, yaw):
    cz = math.cos(yaw * 0.5)
    sz = math.sin(yaw * 0.5)
    cy = math.cos(pitch * 0.5)
    sy = math.sin(pitch * 0.5)
    cx = math.cos(roll * 0.5)
    sx = math.sin(roll * 0.5)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cz * cy - cx * sy * sz
    y = sx * cy * sz + cx * sy * cz
    z = cx * cy * sz - sx * sy * cz
    return x, y, z, w
    #网络不同 输入输出不同
class BJTUDance:
    # 初始化方法
    def __init__(self):
        self.device = torch.device('cpu')  # cpu cuda
        self.num_obs = 60  # 94 63 60 42
        self.num_acts = 18  # 12
        self.scale = {"lin_vel": 2.0,
                      "ang_vel": 0.25,
                      "dof_pos": 1.0,
                      "dof_vel": 0.05,
                      "euler": 1.,
                      "height_measurements": 5.0,
                      "clip_observations": 100.,
                      "clip_actions": 2.5,
                      "clip_arm_actions": 1.2,
                      "action_scale": 0.25}
        default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1., -1.5, -0.1, 1., -1.5, 0, 0, 0, 0, 0, 0, 0, 0]  # LF RF LH RH
        self.default_dof_pos = to_torch(default_dof_pos[0:self.num_acts], device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)
        self.imu_euler = torch.zeros(3, device=self.device, requires_grad=False)
        self.imu_wxyz  = torch.zeros(3, device=self.device, requires_grad=False)

        self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
        self.actions = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)

        self.delay = 0
        print("self.delay: ", self.delay)
        self.action_buf_len = 3
        self.action_history_buf = torch.zeros(self.action_buf_len, self.num_acts, device=self.device, dtype=torch.float)

        p_gains = [150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 20., 15., 10., 10., 10.]
        d_gains = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0.1, 0.1, 0.1, 0.1, 0.1]
        self.p_gains = to_torch(p_gains[0:self.num_acts], device=self.device, requires_grad=False)
        self.d_gains = to_torch(d_gains[0:self.num_acts], device=self.device, requires_grad=False)

        self.joint_qd = np.zeros((4, 3))
        self.joint_arm_d = np.zeros((self.num_acts - 12,))

        self.model_path0 = " "
        self.model_path1 = " "
        self.model_path2 = " "
        self.model_path3 = " "
        self.model_path4 = " "
        self.model_path5 = " "
        self.model_path6 = " "
        self.model_path7 = " "
        self.model_path8 = " "
        self.model_path9 = " "
        self.model_path10 = " "
        self.model_select = 0


    def loadPolicy(self):
        # 测试模型，所有的都可以放进来
        self.model_0 = torch.jit.load(self.model_path0).to(self.device)
        self.model_0.eval()
        self.model_1 = torch.jit.load(self.model_path1).to(self.device)
        self.model_1.eval()
        self.model_2 = torch.jit.load(self.model_path2).to(self.device)
        self.model_2.eval()
        self.model_3 = torch.jit.load(self.model_path3).to(self.device)
        self.model_3.eval()
        self.model_4 = torch.jit.load(self.model_path4).to(self.device)
        self.model_4.eval()
        self.model_5 = torch.jit.load(self.model_path5).to(self.device)
        self.model_5.eval()
        self.model_6 = torch.jit.load(self.model_path6).to(self.device)
        self.model_6.eval()
        self.model_7 = torch.jit.load(self.model_path7).to(self.device)
        self.model_7.eval()
        self.model_8 = torch.jit.load(self.model_path8).to(self.device)
        self.model_8.eval()
        self.model_9 = torch.jit.load(self.model_path9).to(self.device)
        self.model_9.eval()
        self.model_10 = torch.jit.load(self.model_path10).to(self.device)
        self.model_10.eval()
        

    def PutToNet(self):
        x, y, z, w = rpy2quaternion(self.imu_euler[0],
                                    self.imu_euler[1],
                                    self.imu_euler[2])
        base_quat = to_torch([x, y, z, w], dtype=torch.float32, device=self.device).unsqueeze(0)

        base_ang_vel_w = [self.imu_wxyz[0] , self.imu_wxyz[1] , self.imu_wxyz[2]]
        base_ang_vel_w = to_torch(base_ang_vel_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        base_ang_vel = base_ang_vel_w

        gravity_vec = to_torch([0, 0, -1], dtype=torch.float32, device=self.device).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec).squeeze(0)

        self.actor_state[0:3] = base_ang_vel * self.scale["ang_vel"]
        self.actor_state[3:6] = projected_gravity

        self.actor_state[6: 6 + self.num_acts] = (self.dof_pos - self.default_dof_pos) * self.scale["dof_pos"]
        # 60:
        self.actor_state[6 + self.num_acts: 6 + self.num_acts * 2] = self.dof_vel * self.scale["dof_vel"]
        #self.actor_state[6 + self.num_acts: 6 + self.num_acts * 2] = self.dof_vel * self.scale["dof_vel"] * 0  # *0

        self.actor_state[6 + self.num_acts * 2: 6 + self.num_acts * 3] = self.actions



    def inference_(self):
        self.PutToNet()

        with torch.no_grad():
            if(self.model_select == 0):
                actions = self.model_0(self.actor_state)
            if(self.model_select == 1):
                actions = self.model_1(self.actor_state)
            if(self.model_select == 2):
                actions = self.model_2(self.actor_state)
            if(self.model_select == 3):
                actions = self.model_3(self.actor_state)
            if(self.model_select == 4):
                actions = self.model_4(self.actor_state)
            if(self.model_select == 5):
                actions = self.model_5(self.actor_state)
            if(self.model_select == 6):
                actions = self.model_6(self.actor_state)
            if(self.model_select == 7):
                actions = self.model_7(self.actor_state)
            if(self.model_select == 8):
                actions = self.model_8(self.actor_state)
            if(self.model_select == 9):
                actions = self.model_9(self.actor_state)
            if(self.model_select == 10):
                actions = self.model_10(self.actor_state)
            if(self.model_select>10 or self.model_select<0):
                actions = self.model_0(self.actor_state)


        actions.to(self.device)

        # 先存入再裁剪
        self.action_history_buf = torch.cat([self.action_history_buf[1:], actions[None, :]], dim=0)
        actions = self.action_history_buf[-self.delay - 1]
        
        clip_actions = self.scale["clip_actions"] / self.scale["action_scale"]
        clip_arm_actions = self.scale["clip_arm_actions"] / self.scale["action_scale"]

        self.actions[:12] = torch.clip(actions[:12], -clip_actions, clip_actions).to(self.device)
        self.actions[12:] = torch.clip(actions[12:], -clip_arm_actions, clip_arm_actions).to(self.device)

        
        
        actions_scaled = self.actions * self.scale["action_scale"]
        joint_qd = actions_scaled + self.default_dof_pos
        
        print('joint_qd',joint_qd)

        for i in range(4):
            for j in range(3):
                self.joint_qd[i][j] = joint_qd.tolist()[i * 3 + j]

        for i in range(self.num_acts - 12):
            self.joint_arm_d[i] = joint_qd.tolist()[12 + i]
            
        if self.model_select == 5:
            T = time.perf_counter()
            self.joint_qd[0][0] = 0.5
            self.joint_qd[1][0] = -0.5

            self.joint_qd[0][1] = 1 + math.sin(T) * 0.4
            self.joint_qd[1][1] = 1 + math.sin(T) * 0.4

            self.joint_qd[0][2] = -2.16 + math.sin(3 * T) * 0.4
            self.joint_qd[1][2] = -2.16 + math.sin(3 * T) * 0.4
