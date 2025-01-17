import torch
import getsharememory
import ctypes
import BJDance as BJ
import numpy as np
import math
import time
import signal
import yaml


with open('./bjtu_config.yaml', 'r') as file:
    model_path = yaml.safe_load(file)
rocker = [1.3,0.5,0.5]
model_select = 0
model_select_last = 0

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
    #exit(0)

def main():
    global is_running
    global model_select
    global model_select_last

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
        shareinfo_feed = getsharememory.GetFromShareMem(shmaddr,semaphore)
        gait_cmd = shareinfo_feed.ocu_package.gait_info_cmd
        if gait_cmd in cmd_to_model_select:
            BJ_ex.model_select = cmd_to_model_select[gait_cmd]

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
            
        # 打印关节数据（真实/期望） 
        #print("---print start------- real  /  command ---------")  
        #print("abad0: {:.4f} / {:.4f} hip0: {:.4f} / {:.2f} knee0: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_q[0][0], BJ_ex.joint_qd[0][0],
                #shareinfo_feed.sensor_package.joint_q[0][1], BJ_ex.joint_qd[0][1],
                #shareinfo_feed.sensor_package.joint_q[0][2], BJ_ex.joint_qd[0][2]))
        #print("abad1: {:.4f} / {:.4f} hip1: {:.4f} / {:.4f} knee1: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_q[1][0], BJ_ex.joint_qd[1][0],
                #shareinfo_feed.sensor_package.joint_q[1][1], BJ_ex.joint_qd[1][1],
                #shareinfo_feed.sensor_package.joint_q[1][2], BJ_ex.joint_qd[1][2]))
        #print("abad2: {:.4f} / {:.4f} hip2: {:.4f} / {:.4f} knee2: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_q[2][0], BJ_ex.joint_qd[2][0],
                #shareinfo_feed.sensor_package.joint_q[2][1], BJ_ex.joint_qd[2][1],
                #shareinfo_feed.sensor_package.joint_q[2][2], BJ_ex.joint_qd[2][2]))
        #print("abad3: {:.4f} / {:.4f} hip3: {:.4f} / {:.4f} knee3: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_q[3][0], BJ_ex.joint_qd[3][0],
                #shareinfo_feed.sensor_package.joint_q[3][1], BJ_ex.joint_qd[3][1],
                #shareinfo_feed.sensor_package.joint_q[3][2], BJ_ex.joint_qd[3][2]))
        #print("abad_v0: {:.4f} / {:.4f} hip_v0: {:.4f} / {:.4f} knee_v0: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_qd[0][0], 0,
                #shareinfo_feed.sensor_package.joint_qd[0][1], 0,
                #shareinfo_feed.sensor_package.joint_qd[0][2], 0))
        #print("abad_v1: {:.4f} / {:.4f} hip_v1: {:.4f} / {:.4f} knee_v1: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_qd[1][0], 0,
                #shareinfo_feed.sensor_package.joint_qd[1][1], 0,
                #shareinfo_feed.sensor_package.joint_qd[1][2], 0))
        #print("abad_v2: {:.4f} / {:.4f} hip_v2: {:.4f} / {:.4f} knee_v2: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_qd[2][0], 0,
                #shareinfo_feed.sensor_package.joint_qd[2][1], 0,
                #shareinfo_feed.sensor_package.joint_qd[2][2], 0))
        #print("abad_v3: {:.4f} / {:.4f} hip_v3: {:.4f} / {:.4f} knee_v3: {:.4f} / {:.4f}".format(
                #shareinfo_feed.sensor_package.joint_qd[3][0], 0,
                #shareinfo_feed.sensor_package.joint_qd[3][1], 0,
                #shareinfo_feed.sensor_package.joint_qd[3][2], 0))  
        #"-----RPY and omega-----"
        #print("roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, wx {:.4f}, wy: {:.4f}, wz: {:.4f}".format(
              #shareinfo_feed.sensor_package.imu_euler[0], shareinfo_feed.sensor_package.imu_euler[1],
              #shareinfo_feed.sensor_package.imu_euler[2], shareinfo_feed.sensor_package.imu_wxyz[0],
              #shareinfo_feed.sensor_package.imu_wxyz[1], shareinfo_feed.sensor_package.imu_wxyz[2]))
 
        # 打印结束        
        
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
            BJ_ex.inference_()
            for i in range(4):
                for j in range(3):
                    shareinfo_send.tsinghua_rec_package.joint_q_d[i][j] = BJ_ex.joint_qd[i][j]
            # print("qd",BJ_ex.joint_qd)
        getsharememory.PutToShareMem(shareinfo_send,shareinfo_send.tsinghua_rec_package,shmaddr,semaphore)
        elapsed_time = time.perf_counter() - start_time
        # print("current cal time", elapsed_time)
        if elapsed_time < 0.02:
            time.sleep(0.02 - elapsed_time)#保证50Hz频率

    getsharememory.ShareMemClose(shmaddr,semaphore,shmid)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
