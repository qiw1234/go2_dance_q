# -*- coding: utf-8 -*-
import torch
import getsharememory_twodogs
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time
import csv
import os
import threading
import keyboard
import random
from kinematics import *
from copy import deepcopy
import yaml
import ctypes

# model 0: stand
model_path_test0 = './model/go2/stand_2025-03-14_09-07-41.jit' #
# model 1: arm leg
model_path_test1 = './model/test/arm_leg_2025-02-27_21-05-49.jit'
# model 2: wave
model_path_test2 = './model/test/wave_0112_1.jit'  #  input 60 pd150 单臂挥舞 挥舞幅度大
# model 3: trot
model_path_test3 = './model/go2/trot_2025-03-12_09-40-11.jit'
# model 4: swing
model_path_test4 = './model/go2/swing_2025-03-12_09-37-50.jit'
# model 5: turn and jump
model_path_test5 = './model/test/turn_and_jump_0114_1.jit' # 跳跃turn_and_jump_0107_1不行
# model 6: wave two leg 1
model_path_test6 = './model/test/wavetwoleg_model_18000.jit'
# model 7: wave two leg 2
model_path_test7 = './model/test/wavetwoleg2_model_26000.jit'

TEST = False

def s(x):
    return math.sin(x)


def c(x):
    return math.cos(x)


def t(x):
    return math.tan(x)


def limit(a, min_, max_):
    value = min(max(a, min_), max_)
    return value


min_pos = [[-0.69, -0.78, -2.6 - 0.262],
           [-0.87, -0.78, -2.6 - 0.262],
           [-0.69, -0.78, -2.6 - 0.262],
           [-0.87, -0.78, -2.6 - 0.262]]
max_pos = [[0.87, 3.00, -0.45 - 0.262],
           [0.69, 3.00, -0.45 - 0.262],
           [0.87, 3.00, -0.45 - 0.262],
           [0.69, 3.00, -0.45 - 0.262]]

max_effort = [160, 180, 572]
max_vel = [19.3, 21.6, 12.8]
joint_up_limit = [0.69, 3.92, -0.52]
joint_low_limit = [-0.87, -1.46, -2.61]


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


class BJTUDance:
    # 初始化方法
    def __init__(self):
        self.device = torch.device('cpu')  # cpu cuda
        self.num_obs = 42  # 60 94 63
        self.num_acts = 12  # 12
        self.scale = {"lin_vel": 2.0,
                      "ang_vel": 0.25,
                      "dof_pos": 1.0,
                      "dof_vel": 0.05, #0.05
                      "height_measurements": 5.0,
                      "clip_observations": 100.,
                      "clip_actions": 2.5,
                      "clip_arm_actions": 1.2,
                      "action_scale": 0.25}#0.25
        default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]  # LF RF LH RH
        self.default_dof_pos = to_torch(default_dof_pos[0:self.num_acts], device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)

        self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
        self.actions = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(size=(self.num_acts,), device=self.device, requires_grad=False)


        self.delay = 0
        self.action_buf_len = 3
        self.action_history_buf = torch.zeros(self.action_buf_len, self.num_acts, device=self.device, dtype=torch.float)
        self.action_history_buf2 = torch.zeros(self.action_buf_len, self.num_acts, device=self.device,
                                               dtype=torch.float)
        # 越大延迟越高
        self.delay_factor = 0.3

        p_gains = [20] * self.num_acts
        d_gains = [0.5] * self.num_acts
        self.p_gains = to_torch(p_gains[0:self.num_acts], device=self.device, requires_grad=False)
        self.d_gains = to_torch(d_gains[0:self.num_acts], device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_acts, device=self.device, requires_grad=False)

        torque_limits = [23.7, 23.7, 35.55] *4
        self.torque_limits = to_torch(torque_limits[0:self.num_acts], device=self.device, requires_grad=False)
        # print("self.torque_limits: ", self.torque_limits)

        self.joint_qd = np.zeros((4, 3))

        self.foot_pos = np.zeros((4, 3))
        self.stand_height = 0.3

        self.lock = threading.Lock()
        self.shareinfo_tem = 0
        self.shareinfo_feed = getsharememory_twodogs.ShareInfo()
        self._inferenceReady = False
        self.IPC_PROJ_ID = 0x5A0C0001  # key键值  50LEIREN:5A0164C9
        self.SHM_SIZE = 2 * 1024 * 1024  # 共享内存大小
        self.SEM_KEY_ID = 0x5C0C0001  # SEM key键值 50LEIREN:5C0164C9
        self.shmaddr, self.semaphore = 0, 0

        self.shmaddr, self.semaphore = getsharememory_twodogs.CreatShareMem()  # change
        self.shareinfo_feed_send = getsharememory_twodogs.ShareInfo()
        print("sizeof ShareInfo is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo()))
        print("sizeof sensor_package is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo().sensor_package2))
        print("sizeof servo_package is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo().servo_package2))
        print("sizeof ocu_package is :", ctypes.sizeof(getsharememory_twodogs.ShareInfo().ocu_package))

        self.event = threading.Event()
        self.key_pressed = None
        self.swing = 0
        self.model_select = 0

        self.F_dof_pos = torch.zeros_like(self.actor_state[6:12], device=self.device, requires_grad=False)

        # 测试动作参数
        self._targetPos_1 = np.array([  [0.0, 1.36, -2.65], [0.0, 1.36, -2.65],
                                        [-0.2, 1.36, -2.65], [0.2, 1.36, -2.65]])
        self._targetPos_2 = np.array([  [0.0, 0.67, -1.3], [0.0, 0.67, -1.3],
                                        [0.0, 0.67, -1.3], [0.0, 0.67, -1.3]])
        self._targetPos_3 = np.array([  [-0.35, 1.36, -2.65], [0.35, 1.36, -2.65],
                                        [-0.5, 1.36, -2.65], [0.5, 1.36, -2.65]])

        self.startPos = np.zeros((4, 3))
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.firstRun = True

        # 加载模型
        self.loadPolicy()

    def on_key_press(self, event):
        self.key_pressed = event.name
        self.event.set()  # 设置事件，通知主线程处理

    def listen_keyboard(self):
        keyboard.on_press(self.on_key_press)
        while True:
            time.sleep(0.2)  # 模拟长时间运行
            # 都减15°

    def update_keyboard(self):
        if self.event.is_set():
            if self.key_pressed == 'b':
                self.swing = 1
            if self.key_pressed == 'u':
                self.stand_height += 0.01
            if self.key_pressed == 'i':
                self.stand_height -= 0.01
            if self.key_pressed == 'w':
                self.shareinfo_feed_send.ocu_package.x_des_vel += 0.05
            if self.key_pressed == 's':
                self.shareinfo_feed_send.ocu_package.x_des_vel -= 0.05
            if self.key_pressed == 'd':
                self.shareinfo_feed_send.ocu_package.yaw_turn_dot += 0.05
            if self.key_pressed == 'a':
                self.shareinfo_feed_send.ocu_package.yaw_turn_dot -= 0.05
            if self.key_pressed == '0':
                self.model_select = 0
            if self.key_pressed == '1':
                self.model_select = 1
            if self.key_pressed == '2':
                self.model_select = 2
            if self.key_pressed == '3':
                self.model_select = 3
            if self.key_pressed == '4':
                self.model_select = 4
            if self.key_pressed == '5':
                self.model_select = 5
            if self.key_pressed == '6':
                self.model_select = 6
            if self.key_pressed == '7':
                self.model_select = 7
            if self.key_pressed == '8':
                self.model_select = 8
            if self.key_pressed == 'p':
                self.delay_factor +=0.1
            if self.key_pressed == 'l':
                self.delay_factor -=0.1
            # print(f'delay factor:{self.delay_factor}')
                # self.count = 0
            # if self.key_pressed in range(9):
            #     self.actions[:12] = 0
            #     self.action_history_buf[:,:12] = 0
            self.event.clear()  # 重置事件，等待下一个按键

    def update_data(self):
        self.shareinfo_feed = getsharememory_twodogs.GetFromShareMem(
            self.shmaddr, self.semaphore)
        getLegFK(self.shareinfo_feed.sensor_package.joint_q, self.foot_pos)

    def loadPolicy(self):
        # self.model_swing = torch.jit.load(model_path_swing).to(self.device)
        # self.model_turnjump = torch.jit.load(model_path_turnjump).to(self.device)
        # self.model_swing.eval()
        # self.model_turnjump.eval()

        # 测试模型，所有的都可以放进来
        self.model_test0 = torch.jit.load(model_path_test0).to(self.device)
        self.model_test0.eval()
        self.model_test1 = torch.jit.load(model_path_test1).to(self.device)
        self.model_test1.eval()
        self.model_test2 = torch.jit.load(model_path_test2).to(self.device)
        self.model_test2.eval()
        self.model_test3 = torch.jit.load(model_path_test3).to(self.device)
        self.model_test3.eval()
        self.model_test4 = torch.jit.load(model_path_test4).to(self.device)
        self.model_test4.eval()
        self.model_test5 = torch.jit.load(model_path_test5).to(self.device)
        self.model_test5.eval()
        self.model_test6 = torch.jit.load(model_path_test6).to(self.device)
        self.model_test6.eval()
        self.model_test7 = torch.jit.load(model_path_test7).to(self.device)
        self.model_test7.eval()

    def _compute_torques(self, joint_qd):
        # PD controller
        torques = self.p_gains * (joint_qd - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def PutToDrive(self):
        for i in range(4):
            self.shareinfo_feed_send.servo_package.motor_enable[i] = 1
            for j in range(3):
                # self.shareinfo_feed_send.servo_package.kp[i][j] = max_effort[j]*0.8 #self.p_gains[i * 3 + j]
                # self.shareinfo_feed_send.servo_package.kd[i][j] = max_vel[j]*0.1   #self.d_gains[i * 3 + j]
                self.shareinfo_feed_send.servo_package.kp[i][j] = self.p_gains[i * 3 + j]
                self.shareinfo_feed_send.servo_package.kd[i][j] = self.d_gains[i * 3 + j]
                self.shareinfo_feed_send.servo_package.joint_q_d[i][j] = self.joint_qd[i][j]
                # self.shareinfo_feed_send.servo_package.joint_tau_d[i][j] = self.torques[i * 3 + j]
                
        # print(f'action:{self.action_history_buf[-2]}')
        # print(f'des pos:{self.joint_qd}')

        if self.model_select == 8:
            T = time.perf_counter()

            # if self.count==10:
            #    self.F_dof_pos = (self.actor_state[6:12]).clone()
            # self.count = self.count+1

            # temp = [self.shareinfo_feed_send.servo_package.joint_q_d[0][2], self.shareinfo_feed_send.servo_package.joint_q_d[1][2]]
            self.shareinfo_feed_send.servo_package.joint_q_d[0][0] = 0.5
            self.shareinfo_feed_send.servo_package.joint_q_d[1][0] = -0.5

            self.shareinfo_feed_send.servo_package.joint_q_d[0][1] = 1 + math.sin(T)*0.4
            self.shareinfo_feed_send.servo_package.joint_q_d[1][1] = 1 + math.sin(T)*0.4

            self.shareinfo_feed_send.servo_package.joint_q_d[0][2] = -2.16 + math.sin(3*T)*0.4
            self.shareinfo_feed_send.servo_package.joint_q_d[1][2] = -2.16 + math.sin(3*T)*0.4

        getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send, self.shareinfo_feed_send.ocu_package,
                                             self.shmaddr, self.semaphore)
        getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send, self.shareinfo_feed_send.servo_package,
                                             self.shmaddr, self.semaphore)
        getsharememory_twodogs.PutToShareMem(self.shareinfo_feed_send, self.shareinfo_feed_send.servo_package2,
                                             self.shmaddr, self.semaphore)

        # if self.model_select == 8:
        #     self.shareinfo_feed_send.servo_package.joint_q_d[0][2] = temp[0]
        #     self.shareinfo_feed_send.servo_package.joint_q_d[1][2] = temp[1]
                                            

    def PutToNet(self):
        x, y, z, w = rpy2quaternion(self.shareinfo_feed.sensor_package.imu_euler[0],
                                    self.shareinfo_feed.sensor_package.imu_euler[1],
                                    self.shareinfo_feed.sensor_package.imu_euler[2])
        base_quat = to_torch([x, y, z, w], dtype=torch.float32, device=self.device).unsqueeze(0)

        base_ang_vel_w = [self.shareinfo_feed.sensor_package.imu_wxyz[0],
                          self.shareinfo_feed.sensor_package.imu_wxyz[1],
                          self.shareinfo_feed.sensor_package.imu_wxyz[2]]
        base_ang_vel_w = to_torch(base_ang_vel_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        base_ang_vel = base_ang_vel_w.squeeze(0)
        # base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel_w).squeeze(0)

        gravity_vec = to_torch([0, 0, -1], dtype=torch.float32, device=self.device).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec).squeeze(0)

        self.actor_state[0:3] = base_ang_vel * self.scale["ang_vel"]
        self.actor_state[3:6] = projected_gravity

        # print("base_quat: ", base_quat)
        # print("base_ang_vel: ", base_ang_vel)
        # print("projected_gravity: ", projected_gravity)
        # print("actor_state[0:6]: ", self.actor_state[0:6])

        for i in range(4):  # LF RF LH RH
            for j in range(3):
                self.dof_pos[i * 3 + j] = self.shareinfo_feed.sensor_package.joint_q[i][j]
                self.dof_vel[i * 3 + j] = self.shareinfo_feed.sensor_package.joint_qd[i][j]
        # self.dof_vel[:] = 0  #测试用的

        self.actor_state[6: 6 + self.num_acts] = (self.dof_pos - self.default_dof_pos) * self.scale["dof_pos"]
        
        # if self.model_select == 8:
        #     self.actor_state[6:12] = self.F_dof_pos
        #     print(f'count:{self.count}')
        #     print(f'F_dof_pos:{self.F_dof_pos}')

        self.actor_state[6 + self.num_acts: 6 + self.num_acts * 2] = self.dof_vel * self.scale["dof_vel"]
        self.actor_state[6 + self.num_acts * 2: 6 + self.num_acts * 3] = self.actions
        
        # self.actor_state = torch.clip(self.actor_state, -self.scale["clip_observations"], 
        #                               self.scale["clip_observations"]).to(self.device)
        # 机械臂速度设置为0
        # self.actor_state[6 + self.num_acts + 12: 6 + self.num_acts * 2] = 0
        # 测试，手动添加噪声
        # self.actor_state[6:24] += (2 * torch.rand_like(self.actor_state[6:24]) - 1) * 0.1
        # self.actor_state[0] += (2 * torch.rand_like(self.actor_state[0]) - 1) * 0.3
        # self.actor_state[4:6] += (2 * torch.rand_like(self.actor_state[4:6]) - 1) * 0.5

        # print(self.actor_state[6 + self.num_acts * 2: 6 + self.num_acts * 3])

        # self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
        # print("actor_state[6:42]: ", self.actor_state[6:42])

        # print("dof_pos: ", (self.dof_pos - self.default_dof_pos) * self.scale["dof_pos"])
        # print("actor_state[6 : 6+self.num_acts]: ", self.actor_state[6 : 6+self.num_acts])
        # print("dof_vel: ", self.dof_vel * self.scale["dof_vel"])
        # print("actor_state[6+self.num_acts : 6+self.num_acts*2]: ", self.actor_state[6+self.num_acts : 6+self.num_acts*2])
        # print("actions: ", self.actions)
        # print("actor_state[6+self.num_acts : 6+self.num_acts*2]: ", self.actor_state[6+self.num_acts*2 : 6+self.num_acts*3])

    def test_action(self): 
        if self.firstRun:
            for i in range(4):
                for j in range(3):
                    self.startPos[i, j] = self.shareinfo_feed.sensor_package.joint_q[i][j]
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(4):
                for j in range(3):
                    self.joint_qd[i, j] = (1 - self.percent_1) * self.startPos[i, j] + self.percent_1 * self._targetPos_1[i, j]

        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(4):
                for j in range(3):
                    self.joint_qd[i, j] = (1 - self.percent_2) * self._targetPos_1[i, j] + self.percent_2 * self._targetPos_2[i, j]


        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(4):
                for j in range(3):
                    self.joint_qd[i, j] = self._targetPos_2[i, j] 

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(4):
                for j in range(3):
                    self.joint_qd[i, j] = (1 - self.percent_4) * self._targetPos_2[i, j] + self.percent_4 * self._targetPos_3[i, j]


    def inference_(self):
        last_vel = 0
        last_yaw = 0
        counter = 0
        actor_state = []
        torques = []
        base_euler = []

        while True:
            start_RL_Time = time.perf_counter()
            # print(start_RL_Time - start_RL_Time)
            self.update_data()
            self.update_keyboard()

            if (self.shareinfo_feed_send.ocu_package.x_des_vel != last_vel):
                print('x_vel', self.shareinfo_feed_send.ocu_package.x_des_vel)
            if (self.shareinfo_feed_send.ocu_package.yaw_turn_dot != last_yaw):
                print('yaw_dot', self.shareinfo_feed_send.ocu_package.yaw_turn_dot)

            last_vel = self.shareinfo_feed_send.ocu_package.x_des_vel
            last_yaw = self.shareinfo_feed_send.ocu_package.yaw_turn_dot

            self.PutToNet()
                      
            # 将self.actor_state输出成文件
            actor_state.append(self.actor_state.tolist())
            torques.append(self.torques.tolist())
            base_euler.append([self.shareinfo_feed.sensor_package.imu_euler[0],
                               self.shareinfo_feed.sensor_package.imu_euler[1],
                               self.shareinfo_feed.sensor_package.imu_euler[2],])

            if counter == 3000:
                np.savetxt('data/actor_state.csv', np.array(actor_state), delimiter=",")
                np.savetxt('data/torques.csv', np.array(torques), delimiter=",")
                np.savetxt('data/base_euler.csv', np.array(base_euler), delimiter=",")

                print('data has been saved')
            counter += 1


            # self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
            # self.actor_state = torch.ones(size=(self.num_obs,), device=self.device, requires_grad=False)
            # self.actor_state = torch.randn(size=(self.num_obs,), device=self.device, requires_grad=False)

            with torch.no_grad():
                # actions = self.model_turnjump(self.actor_state)
                # actions2 = self.model_turnjump(self.actor_state2)
                # actions = self.model_test(self.actor_state)
                # actions2 = self.model_test(self.actor_state2)
                if self.model_select == 0:
                    actions = self.model_test0(self.actor_state)
                if self.model_select == 1:
                    actions = self.model_test1(self.actor_state)
                if self.model_select == 2:
                    actions = self.model_test2(self.actor_state)
                if self.model_select == 3:
                    actions = self.model_test3(self.actor_state)
                if self.model_select == 4:
                    actions = self.model_test4(self.actor_state)
                if self.model_select == 5:
                    actions = self.model_test5(self.actor_state)
                if self.model_select == 6:
                    actions = self.model_test6(self.actor_state)
                if self.model_select == 7 or self.model_select == 8:
                    actions = self.model_test7(self.actor_state)  
  
            # print("actions: \n", actions)
            # print("self.actor_state: ", self.actor_state)


            actions = actions.to(self.device)

            # if self.model_select != 9:

            # 存入history_buf
            # print("self.action_history_buf1: \n", self.action_history_buf)
            self.action_history_buf = torch.cat([self.action_history_buf[1:], actions[None, :]], dim=0)

            actions = self.action_history_buf[-self.delay - 1]

            # 剪切 
            clip_actions = self.scale["clip_actions"]/self.scale["action_scale"]
            self.actions[:12] = torch.clip(actions[:12], -clip_actions, clip_actions).to(self.device)

            # print("actions: \n", actions)
            # print("self.actions: \n", self.actions)

            # self.actions = self.last_actions * self.delay_factor + self.actions * (1 - self.delay_factor)
            for i in range(4):
                self.actions[3*i] = self.last_actions[3*i] * self.delay_factor + self.actions[3*i] * (1 - self.delay_factor)
            self.last_actions = self.actions.clone()


            actions_scaled = self.actions * self.scale["action_scale"]
            joint_qd = actions_scaled + self.default_dof_pos

            # joint_qd[4,] = 0.8+self.shareinfo_feed_send.ocu_package.x_des_vel



            for i in range(4):
                for j in range(3):
                    self.joint_qd[i][j] = joint_qd.tolist()[i * 3 + j]
            # self.joint_qd[0][1] = -0.6
            self.torques = self._compute_torques(joint_qd).view(self.torques.shape)

            # print("torques: ", self.torques)
            # print("self.actor_state: ", self.actor_state)
            # print("action_scaled: ", actions_scaled)
            # print('des pos:',self.joint_qd) #RF LF RH LH

            if TEST:
                self.test_action()
                print('actions:',self.joint_qd)

            self.PutToDrive()


            last_time = time.perf_counter() - start_RL_Time
            # print("last_time: ", last_time)  # 大概 3 ms
            if (last_time > 0.02):
                print("time over:", time.perf_counter() - start_RL_Time)
            if last_time < 0.02:
                time.sleep(0.02 - last_time)  # 保证50Hz频率


if __name__ == "__main__":
    bjtudance = BJTUDance()
    bjtudance.keyboard_thread = threading.Thread(target=bjtudance.listen_keyboard)
    bjtudance.keyboard_thread.start()
    while (True):
        bjtudance.inference_()