import sysv_ipc
import os
import ctypes

scale_Cur2Tor = [2.6023*0.95,3.4555*0.95,3.7818*0.95] ##scale_Cur2Tor为电机电流与关节扭矩系数，(此系数为50中空电机狗专用)
##即：关节扭矩=scale_Cur2Tor*电机电流，电机电流kp = 关节扭矩kp * scale_Cur2Tor,电机电流kd = 关节速度kd * scale_Cur2Tor

# 定义C语言中的key_t类型

key_t = ctypes.c_int

# 定义C语言中的shmget函数
shmget = ctypes.CDLL(None).shmget
shmget.argtypes = [key_t, ctypes.c_size_t, ctypes.c_int]
shmget.restype = ctypes.c_int
# 定义C语言中的shmat函数
shmat = ctypes.CDLL(None).shmat
shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
shmat.restype = ctypes.c_void_p
# 定义shmdt函数
shmdt = ctypes.CDLL(None).shmdt
shmdt.argtypes = [ctypes.c_void_p]
shmdt.restype = ctypes.c_int
#定义shmctl函数
shmctl = ctypes.CDLL(None).shmctl
shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
shmctl.restype = ctypes.c_int

libc = ctypes.CDLL('libc.so.6')
# 定义 ftok 函数原型
libc.ftok.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.ftok.restype = ctypes.c_int

#******************说明****************
#如只控制电机层，搜索#!!!，#!!!代表电机、传感器相关变量
#如通过201控制层，只需关注 Train_package_rec  Train_package_send

#控制层、底层 代表机器狗主控内程序
#推理层 代表清华推理程序

#结构体Train_package_rec、Train_package_send用于控制层与推理层数据交互，目前已实现推理层获取完期望关节位置 进行共享内存数据交互后 通过控制层送往底层跟踪期望关节位置
#结构体Sensor_Package为底层送往共享内存数据，供读取关节位置、速度(差分获取)、欧拉角、角速度、IMU加速度
#结构体Sensor_Package为底层送往电机数据，直接修改送入共享内存地址后，底层会直接更改电机期望位置、使能状态等
#*************************************

class Train_package_rec(ctypes.Structure):
    _fields_ = [
        ('joint_q_d', ctypes.c_float*3*4),#期望关节位置 推理层送往控制层 以500Hz频率发送到底层(1000hz)
        ('joint_qd_d', ctypes.c_float*3*4),#期望关节速度 目前未使用
        ('heart',ctypes.c_int)
    ]
class Train_package_send(ctypes.Structure):
    _fields_ = [
        ('enter_tsinghua_mode', ctypes.c_ubyte),#控制层发送给推理层，推理层启动推理
        ('first_train', ctypes.c_ubyte * 4)
    ]
# 定义 Sensor_Package 结构体
class Sensor_Package(ctypes.Structure):
    _fields_ = [
        ('joint_q', ctypes.c_float*3*4),#!!!关节位置反馈  第一维为腿顺序：0 1 2 3 -> 左前 右前 左后 右后 ；第二维为关节顺序：0 1 2 -> 侧展 前摆 膝关节。其余和腿&关节相关的二维变量均相同。
        ('joint_qd', ctypes.c_float*3*4),#!!!关节速度反馈，位置差分/dt获得 
        ('joint_tau', ctypes.c_float*3*4),#!!!关节扭矩反馈 
        ('motor_output', ctypes.c_float*3*4),
        ('imu_euler', ctypes.c_float*3),#!!!IMU欧拉角，0 1 2 -> Roll Pitch Yaw
        ('imu_wxyz', ctypes.c_float *3),#!!!IMU角速度 0 1 2 -> wx wy wz  (机器狗机身坐标系 z正方向垂直向上，x正方向指向前，y正方向指向机身左侧)
        ('imu_acc', ctypes.c_float*3),#!!!IMU加速度 0 1 2 -> ddx ddy ddz
        ('body_vel', ctypes.c_float * 2),
        ('ankle_f', ctypes.c_float*4),
        ('is_ready_go', ctypes.c_char),
        ('drive_task_time', ctypes.c_float),
        ('current_send', ctypes.c_float * 3 * 4),
        ('joint_state', ctypes.c_ubyte * 3 * 4),
        ('joint_servo_state', ctypes.c_uint * 3 * 4),
        ('imu_state', ctypes.c_ubyte),
        ('imu_euler_onboard', ctypes.c_float*3),
        ('imu_omega_onboard', ctypes.c_float*3),
        ('imu_acc_onboard', ctypes.c_float*3)
    ]
class Servo_Package_Socket(ctypes.Structure):
    _fields_ = [
        ("tau_d", ctypes.c_float *3*4),
        ("motor_output_v", ctypes.c_float *3*4),
        ("motor_mode", ctypes.c_char *3*4),
        ("is_put_on_ground", ctypes.c_char),
        ("data_params", ctypes.c_float * 156)
    ]
class Servo_Package(ctypes.Structure):
    _fields_ = [
        ("joint_q_d", ctypes.c_float *3*4),#!!!关节期望位置 第一维为腿顺序： 0 1 2 3 -> 左前 右前 左后 右后 ；第二维为关节顺序： 0 1 2 -> 侧展 前摆 膝关节
        ("joint_qd_d", ctypes.c_float *3*4),#!!!关节期望速度
        ("joint_tau_d", ctypes.c_float *3*4),
        ("kp", ctypes.c_float *3*4),#!!!电机电流kp，kp/(scale_Cur2Tor)转换为关节扭矩kp
        ("kd", ctypes.c_float *3*4),#!!!电机电流kd，kd/(scale_Cur2Tor)转换为关节扭矩kd
        ("kv_ff", ctypes.c_float * 3 * 4),
        ("kff", ctypes.c_float * 3 * 4),
        ("kf_p", ctypes.c_float * 3 * 4),
        ("motor_enable", ctypes.c_int * 4),#!!!电机使能标志，赋1后电机上使能，赋0后电机下使能  0 1 2 3 -> 左前 右前 左后 右后
        ("motor_mode", ctypes.c_char *3*4),#!!! 置1
        ("is_put_on_ground", ctypes.c_char),
        ("heartbeat", ctypes.c_int)
]
class Send_Package_Socket(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int * 4),
        ("q", ctypes.c_float *3*4),
        ("q_d", ctypes.c_float *3*4),
        ("f", ctypes.c_float *3*4),
        ("f_d", ctypes.c_float *3*4),
        ("xyz", ctypes.c_float *3*4),
        ("xyz_d", ctypes.c_float *3*4),
        ("fxyz", ctypes.c_float *3*4),
        ("fxyz_d", ctypes.c_float *3*4),
        ("extra", ctypes.c_float * 32),
        ("angle_vel", ctypes.c_float * 3),
        ("body_vel_xyz", ctypes.c_float * 3),
        ("gait_mode", ctypes.c_ubyte),
        ("sdk_mode", ctypes.c_ubyte),
        ("att", ctypes.c_float * 3),
        ("att_rate", ctypes.c_float * 3),
        ("pos_n", ctypes.c_float * 3),
        ("spd_b", ctypes.c_float * 3),
        ("end_pos_nb", ctypes.c_float * 3 * 4),
        ("end_pos_b", ctypes.c_float * 3 * 4),
        ("cpu_rate", ctypes.c_float * 10),
        ("ooda_mode", ctypes.c_int),
        ("power_mode", ctypes.c_int),
        ("gait_state", ctypes.c_ubyte),
    ]

class Ocu_package(ctypes.Structure):##!!!启动dogj_ctrl_interface_jb_jx后，通过app连接后，才能收到手柄摇杆按键数据
    _fields_ = [
        ("yaw_turn_dot", ctypes.c_float),#!!!手柄右下摇杆“左右”大小范围： [左 -0.5 ~ 0.5 右]
        ("x_des_vel", ctypes.c_float),#!!!手柄左上摇杆“前后”大小范围： [前 1.299 ~ -1.3 后]
        ("y_des_vel", ctypes.c_float),#!!!手柄左上摇杆“左右”大小范围： [左 -0.5 ~ 0.5 右]
        ("gait_info_cmd", ctypes.c_uint8),#!!!手柄按键数值，按下一次按键后，该变量值始终为该按键对应值，只有被其他按键顶掉时值才会修改。 
        #按键值(启动时值是255)：上(15) 下(13) 左(17) 右(16) Y(0) X(1) A(2) B(3) LB(7) RB(8) 左上角“两个方框”(9) 右上角“三个竖线” (10)
        ("extern_joy_des", ctypes.c_float),#右侧摇杆 前:最大-100 后:最大100
        ("extern_des", ctypes.c_float), 
        ("extern_value", ctypes.c_float*6),#!!! 右摇杆前后： extern_value[1] 按键值范围[后 -1~1 前] RT:extern_value[5]: 按键值范围 [0.498(按压到极限)~0(未按压)]  LT:extern_value[4] 按键值范围[-0.499(按压到极限)~0(未按压)]
        ("terrain_type", ctypes.c_int),
        ("height_type", ctypes.c_int),
        ("load_type", ctypes.c_int)
    ]

class ShareInfo(ctypes.Structure):
    _fields_ = [
        ("tsinghua_rec_package", Train_package_rec),
        ("tsinghua_send_package", Train_package_send), 
        ("sensor_package", Sensor_Package),
        ("servo_package", Servo_Package),
        ("ocu_package", Ocu_package),
    ]

class shmid_ds_t(ctypes.Structure):
    _fields_ = [
        ("__key", ctypes.c_int),
        ("uid", ctypes.c_uint),
        ("gid", ctypes.c_uint),
        ("cuid", ctypes.c_uint),
        ("cgid", ctypes.c_uint),
        ("mode", ctypes.c_uint),
        ("__seq", ctypes.c_short),
        ("__pad1", ctypes.c_short),
        ("__glibc_reserved1", ctypes.c_ulong),
        ("__glibc_reserved2", ctypes.c_ulong),
        ("shm_segsz", ctypes.c_ulong),
        ("shm_atime", ctypes.c_long),
        ("shm_dtime", ctypes.c_long),
        ("shm_ctime", ctypes.c_long),
        ("shm_cpid", ctypes.c_int),
        ("shm_lpid", ctypes.c_int),
        ("shm_nattch", ctypes.c_ulong),
        ("__glibc_reserved4", ctypes.c_ulong),
        ("__glibc_reserved5", ctypes.c_ulong),
    ]

def CreatShareMem():
    # 定义路径和项目 ID
    path = b'/home'  # 路径必须是字节字符串
    shm_id = 0x5a
    sem_id = shm_id + 2
    shm_key = (libc.ftok(path, shm_id))
    sem_key = (libc.ftok(path, sem_id))
    if shm_key  == -1:
        raise Exception("ftok 生成键值失败")
    print(f"shm_key: {hex(shm_key)}")
    print(f"sem_key: {hex(sem_key)}")
    shm_size = 2*1024*1024
    # 创建信号量对象
    semaphore = sysv_ipc.Semaphore(sem_key, 0o1000, initial_value=1)
    # 获取共享内存的ID
    shmid = shmget(shm_key, shm_size , 0o666 | 0o1000)  # IPC_CREAT = 0o1000
    # print('shmid :',shmid)
    # 将共享内存连接到当前进程的地址空间
    shmaddr = shmat(shmid, 0, 0)
    return shmaddr,semaphore,shmid
def GetFromShareMem(shmaddr,semaphore):

    # 获取信号量（进入临界区）
    try:
        semaphore.acquire()
        result_ptr = ctypes.cast(shmaddr, ctypes.POINTER(ShareInfo))
        # print('joint_q')
        # for i in range(4):
        #     print(result_ptr.contents.sensor_package.joint_q[i][0],
        #             result_ptr.contents.sensor_package.joint_q[i][1],
        #             result_ptr.contents.sensor_package.joint_q[i][2])
        # print('eluer')
        # print(result_ptr.contents.sensor_package.imu_euler[0]*57.2958,
        #             result_ptr.contents.sensor_package.imu_euler[1]*57.2958,
        #             result_ptr.contents.sensor_package.imu_euler[2]*57.2958)
        # print('imu_wxyz :')
        # print(result_ptr.contents.sensor_package.imu_wxyz[0]*57.2958,
        #             result_ptr.contents.sensor_package.imu_wxyz[1]*57.2958,
        #             result_ptr.contents.sensor_package.imu_wxyz[2]*57.2958)
        # print('ocu x y yaw :')
        # print(result_ptr.contents.ocu_package.x_des_vel,
        #         result_ptr.contents.ocu_package.y_des_vel,
        #         result_ptr.contents.ocu_package.yaw_turn_dot,
        #         result_ptr.contents.ocu_package.gait_info_cmd,
        #         result_ptr.contents.ocu_package.extern_value[5])
        return result_ptr.contents
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        semaphore.release()

def PutToShareMem(shareinfo,package,shmaddr,semaphore):
    # 获取信号量（进入临界区）
    semaphore.acquire()
    # 获取package在内存中偏移
    if ctypes.sizeof(package) == ctypes.sizeof(Train_package_rec):
        package.heart = package.heart + 1
    offset = ctypes.addressof(package) - ctypes.addressof(shareinfo)
    try:
    # 写入结构体
        ctypes.memmove(shmaddr+offset, ctypes.addressof(package), ctypes.sizeof(package))
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 释放信号量（离开临界区）
        semaphore.release()
def ShareMemClose(shmaddr,semaphore,shmid):
    shmid_ds = shmid_ds_t()
    shmctl(shmid, 2, ctypes.byref(shmid_ds))
    if shmid_ds.shm_nattch > 1:
        print('shmid_ds nattach ', shmid_ds.shm_nattch)
        # 释放共享内存
        shmdt(shmaddr)
    else:
        # 释放共享内存
        shmdt(shmaddr)
        # 删除共享内存
        shmctl(shmid, 0, 0)
        # 删除信号量
        semaphore.remove()
        print('delete sharememory, delete  semaphore')