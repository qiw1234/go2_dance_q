import time

from unitree_sdk2py.core.channel import  ChannelSubscriber
from unitree_sdk2py.core.channel import  ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.utils.crc import CRC

from .remote_controller import unitreeRemoteController
from .state_machine import stateMachine

HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0
class robotController:
    def __init__(self):
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.remoteController = unitreeRemoteController()
        self.stateMachine = stateMachine()


    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create low state subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # create gamepad commands subscriber #
        self.gamepad_subscriber = ChannelSubscriber("rt/lf/lowstate", LowState_)
        self.gamepad_subscriber.Init(self.GamepadMessageHandler,10)

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

    def InitLowCmd(self):
        #默认参数
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        # 只用前12个，后8个保留
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg:LowState_):
        # 500Hz
        self.low_state = msg

    def GamepadMessageHandler(self, msg:LowState_):
        # 20Hz
        self.low_state = msg
        wireless_remote_data = self.low_state.wireless_remote
        self.remoteController.parse(wireless_remote_data)

    def run(self):
        raise NotImplementedError

    def send_cmd(self, cmd:LowCmd_):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher.Write(cmd)