import torch
import getsharememory
import ctypes
import BJDance as BJ
import numpy as np
import math
import time
import signal
import yaml
import ctypes
import socket
import struct
class Arm_State(ctypes.Structure):
    _fields_ = [
        ("net_control", ctypes.c_int),
        ("joint_q", ctypes.c_float *6),
        ("joint_v", ctypes.c_float *6),
        ("joint_tau", ctypes.c_float *6),
        ("kp_arm", ctypes.c_float *6),
        ("kd_arm", ctypes.c_float *6),
        ("gripper", ctypes.c_int),
    ]
class Arm_Feed(ctypes.Structure):
    _fields_ = [
        ("joint_q", ctypes.c_float *6),
        ("joint_v", ctypes.c_float *6),
        ("joint_tau", ctypes.c_float *6),
        ("gripper", ctypes.c_int),
    ]
init_pos=[0.00134,0.01469,0.01698,-0.01736,0.00248,0.00362]
with open('./bjtu_config.yaml', 'r') as file:
    model_path = yaml.safe_load(file)
rocker = [1.3,0.5,0.5]
model_select = 0
model_select_last = 0

udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

model_paths = [model_path["model"][f"model{i}"] for i in range(11)]
cmd_to_model_select = {
    220: 0,  
    221: 1,  
    222: 2,  
    223: 3,  
    224: 4, 
    225: 5,
    226: 6,
    227: 7,
    228: 8,
    229: 9,
    230: 10, 
}

is_running = 1

def signal_handler(signal, frame):
    global is_running
    is_running = 0
    print('exit')
    udp_server.close()
    #exit(0)

def main():
    global is_running
    global model_select
    global model_select_last

    arm_state = Arm_State()
    arm_state.net_control = 0
    for i in range(6):
        arm_state.joint_q[i] = init_pos[i]
    
    arm_feed = Arm_Feed()
    udp_clinet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data=0
    udp_server.bind(('192.168.1.201', 2236))
    udp_server.setblocking(0)#0 非阻塞 1阻塞

    for index, path in enumerate(model_paths):
        print(f"model{index}_path", path)
    
    BJ_ex = BJ.BJTUDance()
    BJ_ex.model_path0 = model_paths[0]
    BJ_ex.model_path1 = model_paths[1]
    BJ_ex.model_path2 = model_paths[2]
    BJ_ex.model_path3 = model_paths[3]
    BJ_ex.model_path4 = model_paths[4]
    BJ_ex.model_path5 = model_paths[5]
    BJ_ex.model_path6 = model_paths[6]
    BJ_ex.model_path7 = model_paths[7]
    BJ_ex.model_path8 = model_paths[8]
    BJ_ex.model_path9 = model_paths[9]
    BJ_ex.model_path10 = model_paths[10]
    
    BJ_ex.loadPolicy()

    shmaddr,semaphore,shmid = getsharememory.CreatShareMem()
    ii = 0
    shareinfo_send = getsharememory.ShareInfo()
    ctypes.memset(ctypes.addressof(shareinfo_send),0,ctypes.sizeof(shareinfo_send))
    stand_height = 0.52
    shareinfo_feed = getsharememory.GetFromShareMem(shmaddr,semaphore)
    BJ_ex.inference_()

    while is_running == 1:
        start_time = time.perf_counter()  #记录循环开始时间
        try:
            data, address = udp_server.recvfrom(ctypes.sizeof(Arm_Feed))
        except BlockingIOError:
            arm_state.net_control = 0
            data = 0
            pass
        if(arm_state.net_control == 1 or data != 0):
            fmt = '19i'#!i 表示大端字节序 跨架构平台通信中一般统一使用大端字节序 i小端
            arm_feed_tuple = struct.unpack(fmt, data)
            arm_feed.joint_q[:] = arm_feed_tuple[0:6]
            arm_feed.joint_v[:] = arm_feed_tuple[6:12]
            arm_feed.joint_tau[:] = arm_feed_tuple[12:18]
            arm_feed.gripper = arm_feed_tuple[18]
            #print("joint_q", arm_feed.joint_q[0]/10000, arm_feed.joint_q[1]/10000, arm_feed.joint_q[2]/10000, arm_feed.joint_q[3]/10000, arm_feed.joint_q[4]/10000, arm_feed.joint_q[5]/10000)


        shareinfo_feed = getsharememory.GetFromShareMem(shmaddr,semaphore)
        gait_cmd = shareinfo_feed.ocu_package.gait_info_cmd
        if gait_cmd in cmd_to_model_select:
            BJ_ex.model_select = cmd_to_model_select[gait_cmd]
        if BJ_ex.model_select == 6 and shareinfo_feed.tsinghua_send_package.enter_tsinghua_mode:
            arm_state.net_control = 1
        if BJ_ex.model_select != 6:
            arm_state.net_control = 0
        if(model_select_last != BJ_ex.model_select):
            print("using model ",BJ_ex.model_select)
            model_select_last = BJ_ex.model_select

        for i in range(3):
            BJ_ex.imu_euler[i] = shareinfo_feed.sensor_package.imu_euler[i]
            BJ_ex.imu_wxyz[i] = shareinfo_feed.sensor_package.imu_wxyz[i]

        for i in range(4):
            for j in range(3):
                BJ_ex.dof_pos[i*3+j] = shareinfo_feed.sensor_package.joint_q[i][j]
                BJ_ex.dof_vel[i*3+j] = shareinfo_feed.sensor_package.joint_qd[i][j]
        if(arm_state.net_control == 1 or data != 0):
            for i in range(6):
                if i==1:
                    BJ_ex.dof_pos[12+i] = - arm_feed.joint_q[i]/10000
                if i!=1:
                    BJ_ex.dof_pos[12+i] = arm_feed.joint_q[i]/10000
                #TODO vel
            # print("real", arm_feed.joint_q[0]/10000, arm_feed.joint_q[1]/10000, arm_feed.joint_q[2]/10000, arm_feed.joint_q[3]/10000, arm_feed.joint_q[4]/10000, arm_feed.joint_q[5]/10000)

        BJ_ex.inference_()

        print(BJ_ex.actor_state)
        print(BJ_ex.joint_qd)
        print(BJ_ex.joint_arm_d)

        if (not shareinfo_feed.tsinghua_send_package.enter_tsinghua_mode):
            ii=ii+1
            if(ii%100 == 0):
                print("wait for enter tsinghua_mode")
            for i in range(4):
                for j in range(3):
                    shareinfo_send.tsinghua_rec_package.joint_q_d[i][j] = shareinfo_feed.sensor_package.joint_q[i][j]

        if(shareinfo_feed.tsinghua_send_package.enter_tsinghua_mode):
            stand_height = (stand_height + 0.003 * shareinfo_feed.ocu_package.extern_value[5]
                                         + 0.003 * shareinfo_feed.ocu_package.extern_value[4])#RT升高高度  LT降低高度0.003
            if stand_height > 0.58:
                stand_height = 0.58
            if stand_height < 0.15:
                stand_height = 0.15
            
            
            for i in range(4):
                for j in range(3):
                    shareinfo_send.tsinghua_rec_package.joint_q_d[i][j] = BJ_ex.joint_qd[i][j]
            if arm_state.net_control == 1:
                for i in range(6):
                    if(i==1):
                        arm_state.joint_q[i] = - BJ_ex.joint_arm_d[i]
                    if(i!=1):
                        arm_state.joint_q[i] = BJ_ex.joint_arm_d[i]
                print("cmd  ",BJ_ex.joint_arm_d)

            if arm_state.net_control == 0:
                for i in range(6):
                    arm_state.joint_q[i] = init_pos[i]

        getsharememory.PutToShareMem(shareinfo_send,shareinfo_send.tsinghua_rec_package,shmaddr,semaphore)

        send_data = ctypes.string_at(ctypes.addressof(arm_state), ctypes.sizeof(arm_state))
        if(arm_state.joint_q[2]+1.2>0.1):
            udp_clinet.sendto(send_data, ("192.168.1.200", 2233))
        
        elapsed_time = time.perf_counter() - start_time
        # print("current cal time", elapsed_time)
        if elapsed_time < 0.02:
            time.sleep(0.02 - elapsed_time)#保证50Hz频率

    getsharememory.ShareMemClose(shmaddr,semaphore,shmid)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
