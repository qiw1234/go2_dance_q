#ifndef ROBOT_DRIVER_TASK_H
#define ROBOT_DRIVER_TASK_H
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <utilities/PeriodicTask.h>
#include "utilities/SharedMemory.h"
#include "utilities/Timer.h"

#include <mutex>
#include <queue>
#include <utility>
#include <vector>


#include <stdint.h>
//#include <xp_robot.h>
#if _RAISIM_EN
#include "SimulatorMessage.h"
#else
#include "SimUtilities/SimulatorMessage.h"
#endif
using namespace std;

class xpRobotDriTask:public PeriodicTask//, public xpUdpServer
{
public:
    xpRobotDriTask(PeriodicTaskManager*, float, std::string);
    using PeriodicTask::PeriodicTask;
    //using xpUdpServer::xpUdpServer;
    void init() override;
    void run() override;
    void cleanup() override;
    void run_sim();
    int actions_count = 0;
    //void rcv_calback( void *package, uint16_t port ) override;
    Eigen::Array<float,12,1> getTorque(Eigen::Array<float, 12,1> tauDes, Eigen::Array<float,12,1> qd);

    struct LEG_PARAMS
    {
        float Z1;
        float a1;
        float a2;
        float d;
        float a3_x;
        float a3_z;
        float body_l;
        float body_w;
        //Vec3<float> joint_offset[3];
        float  unloading_force;
        float loading_force;
        //add other contents

    }leg_params;
    enum _LEG_TYPE
    {
        F_KNEE,
        H_KNEE
    };
    int leg_type[4]={H_KNEE,H_KNEE,H_KNEE,H_KNEE};
    float offset_q[4];
private:
    ShareMemory *shareMemory;
    //xpUdpClient *udp_client_drv;

#if !_USE_SIM
    xpCan *_xp_can;
#endif
    struct Sensor_Package *sensor_package;
    int hr;
    Timer time_ctrl;
    float last_time = 0.0f;
    float now_time = 0.0f;
    float dt = 0.0f;

    int16_t send_data[5][5];
#if _RAISIM_EN
    SharedMemoryObject<SimulatorSyncronizedMessage> *_sharedMemory;
    SharedMemoryObject<SimulatorSyncronizedMessage> *_sharedMemory2;
#else
    SharedMemoryObject<SimulatorSyncronizedMessage> _sharedMemory;
#endif
    //xpRobot *xp_robot;
    float ini_abad_q[4];
    float ini_hip_q[4];
    float ini_knee_q[4];
#ifdef _USE_ARM
float ini_arm_q[_DOF_ARM];
#endif
#if _TC_EN
float _tauMax = 12.5, _V = 48.0, _gr = 16.227, _kt = 0.281, _R = 0.4554, _damping = 0.01, _dryFriction = 0.2;

Eigen::Array<float,12,1> tau_ff,qd;
#else
float _tauMax = 12.5, _V = 48.0, _gr = 16.227, _kt = 0.281, _R = 0.4554, _damping = 0.01, _dryFriction = 0.2;

#endif
    bool firstRun = true;
};
#endif
