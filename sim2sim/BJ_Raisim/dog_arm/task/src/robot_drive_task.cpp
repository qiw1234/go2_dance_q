#include "robot_drive_task.h"
//#include <xp_robot.h>
#include <comm/xpCommon.h>

//#include <xp_locomotion_gait.h>
#include "orientation_tools.h"


#include <fstream>

#define _USE_SOC 0
#if _USE_SOC
#include "adrc.h"
#endif

#define MAX_I 50.0f //60.0
#define SEND_FLAG 1
#define USING_JOYSTICK 1//0

xpRobotDriTask::xpRobotDriTask(PeriodicTaskManager* manager,
                               float period,
                               std::string name):
PeriodicTask(manager, period, name)
{
    //xpRobot::get_instance()->get_body()->setUpVelEstimator(getPeriod());
    //xp_robot = xpRobot::get_instance();
}

void xpRobotDriTask::init()
{
    //gyro = xpGyro::get_instance();

    shareMemory = ShareMemory::get_instance();
#if !_USE_SIM
    _xp_can = xpCan::get_instance();
#endif

    _sharedMemory = new SharedMemoryObject<SimulatorSyncronizedMessage>();
    _sharedMemory->attach(DEVELOPMENT_SIMULATOR_SHARED_MEMORY_NAME);
    _sharedMemory->get()->init();
#if _TWODOG_EN
    _sharedMemory2 = new SharedMemoryObject<SimulatorSyncronizedMessage>();
    _sharedMemory2->attach(DEVELOPMENT_SIMULATOR_SHARED_MEMORY_NAME2);
    _sharedMemory2->get()->init();
#endif


    for(int i=0;i<4;i++)
    {
        if(leg_type[i] == H_KNEE)
        {
            offset_q[i] = atan2(leg_params.a3_x,leg_params.a3_z);
        }
        else
        {
            offset_q[i] = -atan2(leg_params.a3_x,leg_params.a3_z);
        }
    }
}

void xpRobotDriTask::run()
{
    run_sim();
}



Eigen::Array<float,12,1> xpRobotDriTask::getTorque(Eigen::Array<float, 12,1> tauDes, Eigen::Array<float,12,1> qd) {
  // compute motor torque
  Eigen::Array<float,12,1> tauDesMotor = tauDes / _gr;        // motor torque
  Eigen::Array<float,12,1> iDes = tauDesMotor / _kt ;  // i = tau / KT

  Eigen::Array<float,12,1> bemf = qd * _gr * 0.162338045;       // back emf
  Eigen::Array<float,12,1> vDes = iDes * _R + bemf;          // v = I*R + emf
  Eigen::Array<float,12,1> vActual = vDes.cwiseMin(_V).cwiseMax(-_V);//coerce(vDes, -_V, _V);  // limit to battery voltage
  Eigen::Array<float,12,1>tauActMotor = _kt * (vActual - bemf) / _R;  // tau = Kt * I = Kt * V / R
  Eigen::Array<float,12,1> tauAct = _gr * tauActMotor.cwiseMin(_tauMax).cwiseMax(-_tauMax);//coerce(tauActMotor, -_tauMax, _tauMax);


  return tauAct;
}

void xpRobotDriTask::run_sim()
{
    //update joint data
        float joint_q[12] = {_sharedMemory->get()->simToRobot.spiData.q_abad[0],
                   _sharedMemory->get()->simToRobot.spiData.q_hip[0],
                  _sharedMemory->get()->simToRobot.spiData.q_knee[0] - offset_q[0],
                   _sharedMemory->get()->simToRobot.spiData.q_abad[1],
                                  _sharedMemory->get()->simToRobot.spiData.q_hip[1],
                                 _sharedMemory->get()->simToRobot.spiData.q_knee[1] - offset_q[1],
                   _sharedMemory->get()->simToRobot.spiData.q_abad[2],
                                  _sharedMemory->get()->simToRobot.spiData.q_hip[2],
                                 _sharedMemory->get()->simToRobot.spiData.q_knee[2] - offset_q[2],
                   _sharedMemory->get()->simToRobot.spiData.q_abad[3],
                                  _sharedMemory->get()->simToRobot.spiData.q_hip[3],
                                 _sharedMemory->get()->simToRobot.spiData.q_knee[3] - offset_q[3]};
#if _ARX_ARM_EN
    float joint_arm1_q[6] = {_sharedMemory->get()->simToRobot.spiData.q_arm[0],
                                 _sharedMemory->get()->simToRobot.spiData.q_arm[1],
                                 _sharedMemory->get()->simToRobot.spiData.q_arm[2],
                                 _sharedMemory->get()->simToRobot.spiData.q_arm[3],
                                 _sharedMemory->get()->simToRobot.spiData.q_arm[4],
                                 _sharedMemory->get()->simToRobot.spiData.q_arm[5]};
    float joint_arm2_q[6] = {_sharedMemory2->get()->simToRobot.spiData.q_arm[0],
                                 _sharedMemory2->get()->simToRobot.spiData.q_arm[1],
                                 _sharedMemory2->get()->simToRobot.spiData.q_arm[2],
                                 _sharedMemory2->get()->simToRobot.spiData.q_arm[3],
                                 _sharedMemory2->get()->simToRobot.spiData.q_arm[4],
                                 _sharedMemory2->get()->simToRobot.spiData.q_arm[5]};
    // printf("joint_arm1_q %f %f %f %f %f %f \n",joint_arm1_q[0],joint_arm1_q[1],joint_arm1_q[2],joint_arm1_q[3],joint_arm1_q[4],joint_arm1_q[5]);
#endif

        float joint_qd[12] = {_sharedMemory->get()->simToRobot.spiData.qd_abad[0],
                   _sharedMemory->get()->simToRobot.spiData.qd_hip[0],//bug old:q_hip
                  _sharedMemory->get()->simToRobot.spiData.qd_knee[0],
                   _sharedMemory->get()->simToRobot.spiData.qd_abad[1],
                                  _sharedMemory->get()->simToRobot.spiData.qd_hip[1],
                                 _sharedMemory->get()->simToRobot.spiData.qd_knee[1],
                   _sharedMemory->get()->simToRobot.spiData.qd_abad[2],
                                  _sharedMemory->get()->simToRobot.spiData.qd_hip[2],
                                 _sharedMemory->get()->simToRobot.spiData.qd_knee[2],
                   _sharedMemory->get()->simToRobot.spiData.qd_abad[3],
                                  _sharedMemory->get()->simToRobot.spiData.qd_hip[3],
                                 _sharedMemory->get()->simToRobot.spiData.qd_knee[3]};//bug

        float joint_tau[12] = {_sharedMemory->get()->simToRobot.spiData.tau_abad[0],
                               _sharedMemory->get()->simToRobot.spiData.tau_hip[0],
                              _sharedMemory->get()->simToRobot.spiData.tau_knee[0],
                               _sharedMemory->get()->simToRobot.spiData.tau_abad[1],
                                              _sharedMemory->get()->simToRobot.spiData.tau_hip[1],
                                             _sharedMemory->get()->simToRobot.spiData.tau_knee[1],
                               _sharedMemory->get()->simToRobot.spiData.tau_abad[2],
                                              _sharedMemory->get()->simToRobot.spiData.tau_hip[2],
                                             _sharedMemory->get()->simToRobot.spiData.tau_knee[2],
                               _sharedMemory->get()->simToRobot.spiData.tau_abad[3],
                                              _sharedMemory->get()->simToRobot.spiData.tau_hip[3],
                                             _sharedMemory->get()->simToRobot.spiData.tau_knee[3]};
#if _TWODOG_EN
        float joint_q2[12]
            = {_sharedMemory2->get()->simToRobot.spiData.q_abad[0],
               _sharedMemory2->get()->simToRobot.spiData.q_hip[0],
               _sharedMemory2->get()->simToRobot.spiData.q_knee[0] - offset_q[0],
               _sharedMemory2->get()->simToRobot.spiData.q_abad[1],
               _sharedMemory2->get()->simToRobot.spiData.q_hip[1],
               _sharedMemory2->get()->simToRobot.spiData.q_knee[1] - offset_q[1],
               _sharedMemory2->get()->simToRobot.spiData.q_abad[2],
               _sharedMemory2->get()->simToRobot.spiData.q_hip[2],
               _sharedMemory2->get()->simToRobot.spiData.q_knee[2] - offset_q[2],
               _sharedMemory2->get()->simToRobot.spiData.q_abad[3],
               _sharedMemory2->get()->simToRobot.spiData.q_hip[3],
               _sharedMemory2->get()->simToRobot.spiData.q_knee[3] - offset_q[3]};

        float joint_qd2[12]
            = {_sharedMemory2->get()->simToRobot.spiData.qd_abad[0],
               _sharedMemory2->get()->simToRobot.spiData.qd_hip[0],//bug old:q_hip
               _sharedMemory2->get()->simToRobot.spiData.qd_knee[0],
               _sharedMemory2->get()->simToRobot.spiData.qd_abad[1],
               _sharedMemory2->get()->simToRobot.spiData.qd_hip[1],
               _sharedMemory2->get()->simToRobot.spiData.qd_knee[1],
               _sharedMemory2->get()->simToRobot.spiData.qd_abad[2],
               _sharedMemory2->get()->simToRobot.spiData.qd_hip[2],
               _sharedMemory2->get()->simToRobot.spiData.qd_knee[2],
               _sharedMemory2->get()->simToRobot.spiData.qd_abad[3],
               _sharedMemory2->get()->simToRobot.spiData.qd_hip[3],
               _sharedMemory2->get()->simToRobot.spiData.qd_knee[3]};

        float joint_tau2[12] = {_sharedMemory2->get()->simToRobot.spiData.tau_abad[0],
                               _sharedMemory2->get()->simToRobot.spiData.tau_hip[0],
                               _sharedMemory2->get()->simToRobot.spiData.tau_knee[0],
                               _sharedMemory2->get()->simToRobot.spiData.tau_abad[1],
                               _sharedMemory2->get()->simToRobot.spiData.tau_hip[1],
                               _sharedMemory2->get()->simToRobot.spiData.tau_knee[1],
                               _sharedMemory2->get()->simToRobot.spiData.tau_abad[2],
                               _sharedMemory2->get()->simToRobot.spiData.tau_hip[2],
                               _sharedMemory2->get()->simToRobot.spiData.tau_knee[2],
                               _sharedMemory2->get()->simToRobot.spiData.tau_abad[3],
                               _sharedMemory2->get()->simToRobot.spiData.tau_hip[3],
                               _sharedMemory2->get()->simToRobot.spiData.tau_knee[3]};
#endif
        // float joint_q_new[12],joint_qd_new[12],joint_tau_new[12];
        // comm::convert_joint_ij(joint_q,joint_q_new);
        // comm::convert_joint_ij(joint_qd,joint_qd_new);
        // comm::convert_joint_ij(joint_tau,joint_tau_new);
        //******************

    //update base data
    Vec3<float> omega = _sharedMemory->get()->simToRobot.omega.template cast<float>();
    Quat<float> orientation = _sharedMemory->get()->simToRobot.orientation.template cast<float>();
    Vec3<float> acc = _sharedMemory->get()->simToRobot.acc.template cast<float>();
    Vec3<float> pos = _sharedMemory->get()->simToRobot.pos.template cast<float>();
    Vec3<float> vel = _sharedMemory->get()->simToRobot.vel.template cast<float>();
    Vec3<float> euler = ori::quatToRPY(orientation);
    Vec3<float> velw = comm::Rot_ZYX( euler ) * vel;
    // printf("eluer %f %f %f \n",euler[0],euler[1],euler[2]);
    for(int k=0;k<4;k++)
        shareMemory->shareinfo.sensor_package.imu_orientation[k] = orientation[k];
    for(int j = 0;j<3;j++)
    {
        shareMemory->shareinfo.sensor_package.imu_acc[j] = acc[j];

        shareMemory->shareinfo.sensor_package.imu_wxyz[j] = omega[j];

        shareMemory->shareinfo.sensor_package.imu_euler[j] = euler[j];
        #if 1||USING_SIM_ESTIMATOR
        shareMemory->shareinfo.sensor_package.body_vel[j] = vel[j];
        shareMemory->shareinfo.sensor_package.body_pos[j] = pos[j];
        #endif
    }
// printf("body_vel %f %f %f \n",shareMemory->shareinfo.sensor_package.body_vel[0],shareMemory->shareinfo.sensor_package.body_vel[1],shareMemory->shareinfo.sensor_package.body_vel[2]);

#if _TWODOG_EN
    Vec3<float> omega2 = _sharedMemory2->get()->simToRobot.omega.template cast<float>();
    Quat<float> orientation2 = _sharedMemory2->get()->simToRobot.orientation.template cast<float>();
    Vec3<float> acc2 = _sharedMemory2->get()->simToRobot.acc.template cast<float>();
    Vec3<float> pos2 = _sharedMemory2->get()->simToRobot.pos.template cast<float>();
    Vec3<float> vel2 = _sharedMemory2->get()->simToRobot.vel.template cast<float>();
    Vec3<float> euler2 = ori::quatToRPY(orientation2);
    // printf("eluer2 %f %f %f \n",euler2[0],euler2[1],euler2[2]);
    Vec3<float> velw2 = comm::Rot_ZYX( euler2 ) * vel2;
    for(int k=0;k<4;k++)
        shareMemory->shareinfo.sensor_package2.imu_orientation[k] = orientation2[k];
    for(int j = 0;j<3;j++)
    {
        shareMemory->shareinfo.sensor_package2.imu_acc[j] = acc2[j];

        shareMemory->shareinfo.sensor_package2.imu_wxyz[j] = omega2[j];

        shareMemory->shareinfo.sensor_package2.imu_euler[j] = euler2[j];
#if 1||USING_SIM_ESTIMATOR
        shareMemory->shareinfo.sensor_package2.body_vel[j] = vel2[j];
        shareMemory->shareinfo.sensor_package2.body_pos[j] = pos2[j];
#endif
    }
    // printf("eluer2 %f %f %f \n",shareMemory->shareinfo.sensor_package2.imu_euler[0],shareMemory->shareinfo.sensor_package2.imu_euler[1],shareMemory->shareinfo.sensor_package2.imu_euler[2]);
#endif
    //***********
    //useless
    {
#if _USE_SIM
    float footforce[4][3];
    for(int l=0; l<4; l++)
    {
        int j = 0;
        switch (l) {
        case 0:
            j=1;
            break;
        case 1:
            j=0;
            break;
        case 2:
            j=3;
            break;
        case 3:
            j=2;
            break;
        }
        for(int i=0; i<3; i++)
        {
            footforce[l][i] = 0;//_sharedMemory().simToRobot.spiData.foot_force[j][i];
        }
    }
#endif
    }

#if _ARX_ARM_EN
    for(int i=0;i<6;i++)
    {
#if _TWODOG_EN
        shareMemory->shareinfo.sensor_package2.joint_arm[i] = joint_arm2_q[i];
#endif
        shareMemory->shareinfo.sensor_package.joint_arm[i] = joint_arm1_q[i];
    }
#endif
    for(int i = 0;i<4;i++)
    {
        for(int j = 0;j<3;j++)
        {
            shareMemory->shareinfo.sensor_package.joint_q[i][j] = joint_q[i * 3 + j];
            shareMemory->shareinfo.sensor_package.joint_qd[i][j] = joint_qd[i * 3 + j];
            shareMemory->shareinfo.sensor_package.joint_tau[i][j] = joint_tau[i * 3 + j];


            shareMemory->shareinfo.sensor_package.footforce_r[i][j] = _sharedMemory->get()->simToRobot.spiData.foot_force[i][j];
#if _TWODOG_EN
            shareMemory->shareinfo.sensor_package2.joint_q[i][j] = joint_q2[i * 3 + j];
            shareMemory->shareinfo.sensor_package2.joint_qd[i][j] = joint_qd2[i * 3 + j];
            shareMemory->shareinfo.sensor_package2.joint_tau[i][j] = joint_tau2[i * 3 + j];
            shareMemory->shareinfo.sensor_package2.footforce_r[i][j] = _sharedMemory2->get()->simToRobot.spiData.foot_force[i][j];
#endif

        }
    }
    // memset(&shareMemory->shareinfo.sensor_package,0,sizeof(shareMemory->shareinfo.sensor_package));
    shareMemory->PutToShareMem(&shareMemory->shareinfo.sensor_package, sizeof (shareMemory->shareinfo.sensor_package));
    shareMemory->GetFromShareMem(&shareMemory->shareinfo.servo_package, sizeof (shareMemory->shareinfo.servo_package));



#if _TWODOG_EN
    shareMemory->PutToShareMem(&shareMemory->shareinfo.sensor_package2, sizeof (shareMemory->shareinfo.sensor_package2));
    shareMemory->GetFromShareMem(&shareMemory->shareinfo.servo_package2, sizeof (shareMemory->shareinfo.servo_package2));
// printf("joint_arm1_q %f %f %f %f %f %f \n",shareMemory->shareinfo.sensor_package2.joint_arm[0],shareMemory->shareinfo.sensor_package2.joint_arm[1],
//            shareMemory->shareinfo.sensor_package2.joint_arm[2],shareMemory->shareinfo.sensor_package2.joint_arm[3],
//            shareMemory->shareinfo.sensor_package2.joint_arm[4],shareMemory->shareinfo.sensor_package2.joint_arm[5]);
#endif

#if _ARX_ARM_EN
    for(int i=0;i<8;i++)
    {
        _sharedMemory->get()->robotToSim.spiCommand.q_des_arm[i] = shareMemory->shareinfo.servo_package.joint_arm_d[i];
        _sharedMemory2->get()->robotToSim.spiCommand.q_des_arm[i] = shareMemory->shareinfo.servo_package2.joint_arm_d[i];
        _sharedMemory->get()->robotToSim.spiCommand.kp_arm[i] = shareMemory->shareinfo.servo_package.kp_arm[i];
        _sharedMemory2->get()->robotToSim.spiCommand.kp_arm[i] = shareMemory->shareinfo.servo_package2.kp_arm[i];
        _sharedMemory->get()->robotToSim.spiCommand.kd_arm[i] = shareMemory->shareinfo.servo_package.kd_arm[i];
        _sharedMemory2->get()->robotToSim.spiCommand.kd_arm[i] = shareMemory->shareinfo.servo_package2.kd_arm[i];
    }
#endif
    for(int i=0;i<4;i++)
    {
        _sharedMemory->get()->robotToSim.spiCommand.q_des_abad[i] = shareMemory->shareinfo.servo_package.joint_q_d[i][0];
        _sharedMemory->get()->robotToSim.spiCommand.q_des_hip[i] = shareMemory->shareinfo.servo_package.joint_q_d[i][1];
        _sharedMemory->get()->robotToSim.spiCommand.q_des_knee[i] = shareMemory->shareinfo.servo_package.joint_q_d[i][2] + offset_q[i];

        _sharedMemory->get()->robotToSim.spiCommand.qd_des_abad[i] = shareMemory->shareinfo.servo_package.joint_qd_d[i][0];
        _sharedMemory->get()->robotToSim.spiCommand.qd_des_hip[i] = shareMemory->shareinfo.servo_package.joint_qd_d[i][1];
        _sharedMemory->get()->robotToSim.spiCommand.qd_des_knee[i] = shareMemory->shareinfo.servo_package.joint_qd_d[i][2];

        _sharedMemory->get()->robotToSim.spiCommand.kp_abad[i] = shareMemory->shareinfo.servo_package.kp[i][0];
        _sharedMemory->get()->robotToSim.spiCommand.kp_hip[i] = shareMemory->shareinfo.servo_package.kp[i][1];
        _sharedMemory->get()->robotToSim.spiCommand.kp_knee[i] = shareMemory->shareinfo.servo_package.kp[i][2];

        _sharedMemory->get()->robotToSim.spiCommand.kd_abad[i] = shareMemory->shareinfo.servo_package.kd[i][0];
        _sharedMemory->get()->robotToSim.spiCommand.kd_hip[i] = shareMemory->shareinfo.servo_package.kd[i][1];
        _sharedMemory->get()->robotToSim.spiCommand.kd_knee[i] = shareMemory->shareinfo.servo_package.kd[i][2];

        _sharedMemory->get()->robotToSim.spiCommand.tau_abad_ff[i] = shareMemory->shareinfo.servo_package.joint_tau_d[i][0];
        _sharedMemory->get()->robotToSim.spiCommand.tau_hip_ff[i] = shareMemory->shareinfo.servo_package.joint_tau_d[i][1];
        _sharedMemory->get()->robotToSim.spiCommand.tau_knee_ff[i] = shareMemory->shareinfo.servo_package.joint_tau_d[i][2];

#if _TWODOG_EN
        _sharedMemory2->get()->robotToSim.spiCommand.q_des_abad[i] = shareMemory->shareinfo.servo_package2.joint_q_d[i][0];
        _sharedMemory2->get()->robotToSim.spiCommand.q_des_hip[i] = shareMemory->shareinfo.servo_package2.joint_q_d[i][1];
        _sharedMemory2->get()->robotToSim.spiCommand.q_des_knee[i] = shareMemory->shareinfo.servo_package2.joint_q_d[i][2] + offset_q[i];

        _sharedMemory2->get()->robotToSim.spiCommand.qd_des_abad[i] = shareMemory->shareinfo.servo_package2.joint_qd_d[i][0];
        _sharedMemory2->get()->robotToSim.spiCommand.qd_des_hip[i] = shareMemory->shareinfo.servo_package2.joint_qd_d[i][1];
        _sharedMemory2->get()->robotToSim.spiCommand.qd_des_knee[i] = shareMemory->shareinfo.servo_package2.joint_qd_d[i][2];

        _sharedMemory2->get()->robotToSim.spiCommand.kp_abad[i] = shareMemory->shareinfo.servo_package2.kp[i][0];
        _sharedMemory2->get()->robotToSim.spiCommand.kp_hip[i] = shareMemory->shareinfo.servo_package2.kp[i][1];
        _sharedMemory2->get()->robotToSim.spiCommand.kp_knee[i] = shareMemory->shareinfo.servo_package2.kp[i][2];

        _sharedMemory2->get()->robotToSim.spiCommand.kd_abad[i] = shareMemory->shareinfo.servo_package2.kd[i][0];
        _sharedMemory2->get()->robotToSim.spiCommand.kd_hip[i] = shareMemory->shareinfo.servo_package2.kd[i][1];
        _sharedMemory2->get()->robotToSim.spiCommand.kd_knee[i] = shareMemory->shareinfo.servo_package2.kd[i][2];

        _sharedMemory2->get()->robotToSim.spiCommand.tau_abad_ff[i] = shareMemory->shareinfo.servo_package2.joint_tau_d[i][0];
        _sharedMemory2->get()->robotToSim.spiCommand.tau_hip_ff[i] = shareMemory->shareinfo.servo_package2.joint_tau_d[i][1];
        _sharedMemory2->get()->robotToSim.spiCommand.tau_knee_ff[i] = shareMemory->shareinfo.servo_package2.joint_tau_d[i][2];
#endif

    }





}

void xpRobotDriTask::cleanup()
{

}
