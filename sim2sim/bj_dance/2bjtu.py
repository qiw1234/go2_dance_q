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

class Arm_State(ctypes.Structure):
    _fields_ = [
        ("joint_q", ctypes.c_float *6),
        ("joint_v", ctypes.c_float *6),
        ("joint_tau", ctypes.c_float *6),
        ("kp_arm", ctypes.c_float *6),
        ("kd_arm", ctypes.c_float *6),
        ("gripper", ctypes.c_int),
    ]
init_pos=[0.00134,0.01469,0.01698,-0.01736,0.00248,0.00362]
is_running = 1
is_sending = 1
def signal_handler(signal, frame):
    global is_running
    global is_sending
    is_sending = 0
    is_running = 0
    print('exit')
    #exit(0)

def main():
    global is_running
    global model_select
    global model_select_last
    arm_state = Arm_State()
    for i in range(6):
        arm_state.joint_q[i]=0.0
        arm_state.joint_v[i]=0.0
        arm_state.joint_tau[i]=0.0
        arm_state.kp_arm[i]=0.0
        arm_state.kd_arm[i]=0.0
    arm_state.gripper = 0

    for j in range(6):
        arm_state.joint_q[j]=init_pos[j]

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while(1):
        send_data = ctypes.string_at(ctypes.addressof(arm_state), ctypes.sizeof(arm_state))
        arm_state.joint_q[0] = -0.2
        print("joint_q[5] ",arm_state.joint_q[5])
        udp_socket.sendto(send_data, ("192.168.1.200", 2233))
        time.sleep(0.02)
        if is_sending == 0:
            break

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
