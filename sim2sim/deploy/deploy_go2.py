from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import sys
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from remote_controller import unitreeRemoteController
from state_machine import stateMachine, STATES
# from test_user_ctrl import userController
from user_ctrl import userController

PosStopF = 2.146e9
VelStopF = 16000.0


class robotController:
    def __init__(self):
        # 用户控制器
        self.userController = userController()
        # 遥控器
        self.remote_controller = unitreeRemoteController()
        # 状态机
        self.stateMachine = stateMachine()
        # 移动参数
        self.startPos = [-0.03170794,  1.3318926,  -2.7798176,   0.03267688,  1.3222644,  -2.7878265,
                        -0.27484167,  1.3498114,  -2.8058195,   0.2809577,   1.3450881,  -2.8097687 ]

        num_actions = self.userController.num_actions
        # Initializing process variables
        self.qj = np.zeros(num_actions, dtype=np.float32)
        self.dqj = np.zeros(num_actions, dtype=np.float32)
        self.p_gains = self.userController.p_gains
        self.d_gains = self.userController.d_gains
        self.control_dt = self.userController.dt

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        # Initialize the command msg
        self.init_cmd()

        self.lowcmd_publisher_ = ChannelPublisher('rt/lowcmd', LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber('rt/lowstate', LowState_)
        self.lowstate_subscriber.Init(self.LowStateGo2Handler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # 保存数据变量
        self.savedT = []
        self.savedActorState = []
        self.savedCmd = []
        self.startTime = time.perf_counter()

        # 关闭自带的运动服务
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        # 关闭与运动相关的服务
        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def init_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateGo2Handler(self, msg: LowState_):
        self.low_state = msg
        # print('###########################state####################################')
        # print(f'RF HIP:{self.low_state.motor_state[0].q}')
        self.remote_controller.parse(self.low_state.wireless_remote)
        # pos and vel
        for i in range(12):
            self.userController.qj[i] = self.low_state.motor_state[i].q
            self.userController.dqj[i] = self.low_state.motor_state[i].dq
        # imu_state quaternion: w, x, y, z
        self.userController.quat = np.array(self.low_state.imu_state.quaternion)
        self.userController.rpy = np.array(self.low_state.imu_state.rpy)
        self.userController.ang_vel = np.array(self.low_state.imu_state.gyroscope)

    def send_cmd(self, cmd: LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        # print('###########################cmd####################################')
        # print(f'RF cmd :{cmd.motor_cmd[0].q}')
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.Start != 1:
            self.create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def create_zero_cmd(self, cmd):
        size = len(cmd.motor_cmd)
        for i in range(size):
            cmd.motor_cmd[i].q = 0
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0

    def create_damping_cmd(self, cmd):
        size = len(cmd.motor_cmd)
        for i in range(size):
            cmd.motor_cmd[i].q = 0
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 8
            cmd.motor_cmd[i].tau = 0

    def move_to_default_pos(self, kp=60, kd=5):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.control_dt)

        # record the current pos
        init_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.low_state.motor_state[i].q
        # print(f'init dof pos :{init_dof_pos}')
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + self.startPos[j] * alpha
                self.low_cmd.motor_cmd[j].dq = 0
                self.low_cmd.motor_cmd[j].kp = kp
                self.low_cmd.motor_cmd[j].kd = kd
                self.low_cmd.motor_cmd[j].tau = 0
            # print(f'RF HIP:{self.low_state.motor_state[0].q}')
            # print(f'RF cmd :{self.low_cmd.motor_cmd[0].q}')
            # print(f'RF hip error: {self.low_state.motor_state[0].q - self.low_cmd.motor_cmd[0].q}')
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self, kp=60, kd=5):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.A != 1:
            for j in range(12):
                self.low_cmd.motor_cmd[j].q = self.startPos[j]
                self.low_cmd.motor_cmd[j].dq = 0
                self.low_cmd.motor_cmd[j].kp = kp
                self.low_cmd.motor_cmd[j].kd = kd
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def update_state_machine(self):
        if self.remote_controller.B == 1:
            if self.stateMachine.Stop():
                print('damping!!')
        if self.remote_controller.X == 1:
            if self.stateMachine.Defualt():
                print('defualt!!')
        if self.remote_controller.Y == 1:
            if self.stateMachine.Ctrl():
                print('crtl!!')
        if self.remote_controller.Up == 1:
            self.userController.model_select = 0  # stand
        if self.remote_controller.Down == 1:
            self.userController.model_select = 1  # wave
        if self.remote_controller.Left == 1:
            self.userController.model_select = 2  # trot
        if self.remote_controller.Right == 1:
            self.userController.model_select = 3  # swing

    def run(self):
        start_RL_Time = time.perf_counter()
        self.update_state_machine()
        if self.stateMachine.state == STATES.defualt:
            self.move_to_default_pos()
        if self.stateMachine.state == STATES.damp:
            self.create_damping_cmd(self.low_cmd)
            # send the command
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
        if self.stateMachine.state == STATES.ctrl:
            self.userController.inference()
            self.update_low_cmd()
            # send the command
            self.send_cmd(self.low_cmd)
            # 记录数据
            self.savedT.append(time.perf_counter() - self.startTime)
            self.savedActorState.append(self.userController.actor_state.tolist())
            self.savedCmd.append([self.low_cmd.motor_cmd[i].q for i in range(12)])
            # 保证50Hz频率
            last_time = time.perf_counter() - start_RL_Time
            # if last_time > self.control_dt:
            #     print("time over:", time.perf_counter() - start_RL_Time)
            if last_time < self.control_dt:
                time.sleep(self.control_dt - last_time)

    def update_low_cmd(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.userController.des_joint_pos[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.userController.p_gains
            self.low_cmd.motor_cmd[i].kd = self.userController.d_gains
            self.low_cmd.motor_cmd[i].tau = 0

if __name__ == "__main__":

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    # Initialize DDS communication
    ChannelFactoryInitialize(0, sys.argv[1])

    controller = robotController()

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    print("start ctrl loop")
    while True:

        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.Select == 1:
                break
        except KeyboardInterrupt:
            break  # Enter the damping state
    # 保存数据
    T = np.array(controller.savedT)
    actorState = np.array(controller.savedActorState)
    cmd = np.array(controller.savedCmd)
    np.savetxt('data/T.csv', T, delimiter=",")
    np.savetxt('data/actorState.csv', actorState, delimiter=",")
    np.savetxt('data/cmd.csv', cmd, delimiter=",")
    print("data has been saved!")

    controller.move_to_default_pos()
    print("Exit")
