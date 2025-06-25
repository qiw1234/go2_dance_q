#ifndef __SHAREMEMORY_H__
#define __SHAREMEMORY_H__

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
//#include <semaphore.h>
#include <iostream>
#include <string.h>

#include "utilities/sem_com.h"
#if !_RAISIM_EN
#include "/home/ixxuan/Project/ocs2/DogBrainCoreControl_blackdog_sim_panda/sim/include/comm_sim/SimUtilities/SpineBoard.h"
#endif
#define SHAREMEMORY_NAME "xpEDogShareMemory"
#define SHAREMEMORY_MAX_SIZE 2*1024*1024  //2MB MAX_SIZE
#define IPC_PROJ_ID  0x5a

struct Sensor_Package
{
    float joint_q[4][3];
    float joint_qd[4][3];
    float joint_tau[4][3];
#if _ARX_ARM_EN
    float joint_arm[8];
    float joint_arm_dq[8];
#endif

    float motor_output[4][3];

    float imu_euler[3];
    float imu_wxyz[3];
    float imu_acc[3];
    float imu_orientation[4];

    float body_vel[3];
    float body_pos[3];

    float ankle_f[4];
    char is_ready_go;

    float footforce_r[4][3];

};

struct Servo_Package_Socket
{
    float tau_d[4][3];
    float motor_output_v[4][3];
    char  motor_mode[4][3];
    char  is_put_on_ground;

    float data_params[156];
};
struct Servo_Package
{
    // float actions[4][3];
    float joint_q_d[4][3];
    float joint_qd_d[4][3];
    float joint_tau_d[4][3];
#if _ARX_ARM_EN
    float joint_arm_d[8];
#endif
#if _ARX_ARM_EN
    float kp_arm[8];
    float kd_arm[8];
#endif
    float kp[4][3];
    float kd[4][3];

    int motor_enable[4];
    char  motor_mode[4][3];
    char  is_put_on_ground;
    int cmd_info;
};

struct Send_Package_Socket
{
    int mode[4];
    float q[4][3];
    float q_d[4][3];
    float f[4][3];
    float f_d[4][3];
    float xyz[4][3];
    float xyz_d[4][3];
    float fxyz[4][3];
    float fxyz_d[4][3];

    float extra[32];
};

#pragma pack(1)
struct Send_Package_Socket_Show
{
    float angle_expect[4][3];
    float angle_actual[4][3];
    float torque_expect[4][3];
    float torque_actual[4][3];
    float imu_angle[3];
    float electricity;
    short gait;
    short gait_state;
    short ctrl_mode;
    float vel;
    float pos_target[2];
    float pos_person[2];
    unsigned char flag_map;
    unsigned char sensor_state[4][3];
    unsigned char imu_state;
};
#pragma pack()

struct Ocu_package
{
    float yaw_turn_dot;//4
    float x_des_vel;//4
    float y_des_vel;//4
    uint8_t gait_info_cmd;
    float kp;
    float kd;
    uint8_t update_count;
};

struct Servo_Package_Test
{
    float joint_q_d[4][3];
    float joint_qd_d[4][3];
    float joint_tau_d[4][3];

    float kp[4][3];
    float kd[4][3];
    float kff[4][3];
    int motor_enable[4];
    char  motor_mode[4][3];
    char  is_put_on_ground;
};

struct NMPC_Cmd_Input
{
    float vel_cmd[6];//x y z yaw pitch roll  TIME_TO_TARGET=MPC.timeHorizon
    float pos_cmd[6];//x y z yaw pitch roll
    float target_rot_vel;//TARGET_ROTATION_VELOCITY
    float target_dis_vel;//TARGET_DISPLACEMENT_VELOCITY
    uint8_t gait_cmd;
    uint8_t ocu_cmd_type;
    uint8_t cmd_ID;
    uint8_t traj_points_num;
    float time_stamp[100];//time_stamp of the point
    float traj_vel[100][6];//x y z yaw pitch roll
    float traj_pos[100][6];//x y z yaw pitch roll
    float traj_joint[100][12];//joint
};

struct NMPC_State_Input
{
    float joint_q[4][3];
    float joint_qd[4][3];
    float joint_tau[4][3];
    float motor_output[4][3];
    float imu_orientation[4];
    float imu_wxyz[3];
    float imu_acc[3];
    float footforce[4][3];
    int contact_state[4];

};

struct NMPC_Ctrl_Output
{
    float joint_q_d[4][3];
    float joint_qd_d[4][3];
    float joint_tau_d[4][3];
    float kp[4][3];
    float kd[4][3];
    uint8_t mpc_state;
};
struct NMPC_State_Output
{
    float robot_pos[6];
    float robot_vel[6];
};

struct BJTU_Cmd_Input
{
    float vel_cmd[6];//x y z yaw pitch roll  TIME_TO_TARGET=MPC.timeHorizon
    float pos_cmd[6];//x y z yaw pitch roll
    uint8_t gait_cmd;//0~10: stand, standtrot, walk, trot, flytrot, pace, bound, Jump, land PDjump end&savedata
    bool startMPC;//启动MPC计算
};

struct BJTU_State_Input
{
    float joint_q[4][3];//关节角度 髋连杆水平、大小腿伸直为0位  // RF RH LF LH
    float joint_qd[4][3];//关节角速度 未使用
    float joint_tau[4][3];//关节扭矩
    float imu_euler[3];//IMU输出的躯干欧拉角（W系，与切换时的Body系方向一致）roll pitch yaw
    float imu_wxyz[3];//陀螺仪测量的躯干角速度（Body系） x y z
    float imu_acc[3];//加速度计测量的躯干线加速度（Body系）x y z
    float footforce[4][3];//地面反力（Body系）RF RH LF LH

    float body_pos[3];
    float body_vel[3];

};

struct BJTU_Ctrl_Output
{
    float joint_q_d[4][3];  // RF RH LF LH
    float joint_qd_d[4][3];
    float joint_tau_d[4][3];
    float kp[4][3];
    float kd[4][3];
    uint8_t mpc_state;
};



struct ShareInfo
{
    Sensor_Package sensor_package;
#if _TWODOG_EN
    Sensor_Package sensor_package2;
#endif

    Servo_Package servo_package;
#if _TWODOG_EN
    Servo_Package servo_package2;
#endif
    Ocu_package ocu_package;
    Servo_Package_Socket servo_package_socket;
    Send_Package_Socket send_package_socket;//foot?
    Servo_Package_Test servo_package_test;
    NMPC_Cmd_Input nmpc_cmd_input;
    NMPC_State_Input nmpc_state_input;
    NMPC_Ctrl_Output nmpc_ctrl_output;
    NMPC_State_Output nmpc_state_output;

    BJTU_Cmd_Input bjtu_cmd_input;
    BJTU_State_Input bjtu_state_input;
    BJTU_Ctrl_Output bjtu_ctrl_output;
#if !_RAISIM_EN
    SpiCommand spiCommand;
#endif
    //Send_Package_Socket_Show send_package_socket_show;
};

struct ocu_info
{
    float data_from_socket[156];
};
class ShareMemory
{
public:
    static ShareMemory *get_instance(const char *sharememory_name = SHAREMEMORY_NAME);
    ShareMemory(const char *name = SHAREMEMORY_NAME);
    ~ShareMemory() {ShareMemClose(true);}
    int PutToShareMem(void *src_buffer, unsigned int length);
    int GetFromShareMem(void *dts_buffer, unsigned int length);
    void ShareMemClose(bool bDestroyShm = false);
    ShareInfo shareinfo;
private:
    void *CreateShareMemory(const char *name, unsigned int size);

    void SemLock ( void )
    {
        m_sem->sem_p();
    }

    void SemUnLock ( void )
    {
        m_sem->sem_v();
    }

    unsigned int GetOffset(void *member)
    {
        unsigned int offset = 0;
        offset = (unsigned long)(member) - (unsigned long)(&(shareinfo));
        return offset;
    }

    int creat_keyid(const char *name, int m_id)
    {
        int key_id;
        key_id = ftok("/home", m_id);//12345678;//ftok(".", m_id);
        printf("%s key_id:%X\n", name, key_id);
        return key_id;
    }
    sem_com *m_sem;
    int shmid;
    void *m_shareMemory;
};

#endif
