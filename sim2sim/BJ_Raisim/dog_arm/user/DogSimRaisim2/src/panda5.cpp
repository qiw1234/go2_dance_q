// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#include <fstream>
#include<stdio.h>
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "raisim/object/ArticulatedSystem/ArticulatedSystem.hpp"
#include "../../../sim/include/Raisim/SimulatorMessage.h"
#include "../../../common/include/cppTypes.h"
#include "../../../common/include/orientation_tools.h"
#include "../../../Dynamics_party/CommonWbc/include/Dynamics/spatial.h"
#include "../../../Dynamics_party/CommonWbc/include/Dynamics/ActuatorModel.h"
#include <utilities/SharedMemory.h>
#include<math.h>
#if WIN32
#include <timeapi.h>
#endif

std::mutex _robotMutex;
SharedMemoryObject<SimulatorSyncronizedMessage> _sharedMemory,_sharedMemory2;
Eigen::VectorXd pos,orient,vel,omega,acc,gc,gv,tau_now,pos2,orient2,vel2,omega2,acc2,gc2,gv2,tau_now2;
Eigen::VectorXd tau_arm,tau_arm2;
Eigen::Matrix<float,4,1> quad_o;
raisim::ArticulatedSystem *panda5,*panda7;
Vec3<float> foot_force[4],foot_force2[4];
double roll,pitch,yaw,roll2,pitch2,yaw2;
raisim::Box *ditch;
std::vector<ActuatorModel<double>> _actuatorModels;
SVec<float> last_vel,last_vel2;
int dt_i = 1;
int num_dof = 26;
#if _ARX_ARM_EN
#define USE_ARM_DOF = 18;
#endif
const char* datapath= "/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/raisim2.txt";
int dof_root = 6;//xyz eluer
int dof_leg = 12;
int dof_arm = 8;
std::ofstream datafile;
/*
输入：x,y,z,w　为四元数
输出：roll，pitch，yaw欧拉角
**/
static void toEulerAngle( double x, double y, double z, double w, double& roll, double& pitch, double& yaw)
{
    // roll (x-axis rotation)
    double sinr_cosp = +2.0 * (w * x + y * z);
    double cosr_cosp = +1.0 - 2.0 * (x * x + y * y);
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = +2.0 * (w * y - z * x);
    if (fabs(sinp) >= 1)
        pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        pitch = asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = +2.0 * (w * z + x * y);
    double cosy_cosp = +1.0 - 2.0 * (y * y + z * z);
    yaw = atan2(siny_cosp, cosy_cosp);
    //    return yaw;
}
void FK(Eigen::VectorXd q, Eigen::VectorXd tau)
{

    // float k_leg_flag_w = 1.0;
    // float a1 = 0;
    // float a2 = 0.340;
    // float d = 0.119125;//0.10575;//0.119125;
    // float a3 = 0.38237;//sqrt(0.37151*0.37151+0.01478*0.01478);//0.38237;

    //*********70_oldleg_new*********
    float k_leg_flag_w = 1.0;
    float a1 = 0;
    float a2 = 0.340;
    float d = 0.119875;//0.10575;//0.119125;
    float a3 = 0.34;//0.37180;//0.37180 sqrt(0.37151*0.37151+0.01478*0.01478);//0.38237;
    Mat4<float> tansform_mat_d;
    Mat3<float> jacobian_mat;
    Vec3<float> m_f;

    for(int leg = 0; leg<4; leg++)
    {
        float C1 = cos(q[leg*3+0]);
        float S1 = sin(q[leg*3+0]);
        float C2 = cos(q[leg*3+1]);
        float S2 = sin(q[leg*3+1]);
        float C23 = cos(q[leg*3+1]+q[leg*3+2]);
        float S23 = sin(q[leg*3+1]+q[leg*3+2]);
        if(leg==1||leg==3) k_leg_flag_w*=-1.0f;//

        jacobian_mat(0,0) = 0.0;
        jacobian_mat(0,1) =-(a2*C2+a3*C23);
        jacobian_mat(0,2) =-a3*C23;

        jacobian_mat(1,0) = C1*(a1+a2*C2+a3*C23) - d * k_leg_flag_w* S1;
        jacobian_mat(1,1) = -S1*(a2*S2+a3*S23);
        jacobian_mat(1,2) = -a3*S1*S23;

        jacobian_mat(2,0) = S1*(a1+a2*C2+a3*C23)+ d * k_leg_flag_w* C1;
        jacobian_mat(2,1) = C1*(a2*S2+a3*S23);
        jacobian_mat(2,2) = a3*C1*S23;

        m_f[0] = -tau[leg*3+0];
        m_f[1] = -tau[leg*3+1];
        m_f[2] = -tau[leg*3+2];

        foot_force[leg] = jacobian_mat.transpose().inverse() * m_f;
        //printf("#####joint_force######%f %f %f\n",m_f[0],m_f[1],m_f[2]);

        //printf("#####foot_force######%f %f %f\n",foot_force[leg][0],foot_force[leg][1],foot_force[leg][2]);
    }

}
void extern_force()
{
    size_t dj_id = panda5->getBodyIdx ("base");
    raisim::Vec<3> exforce;
    raisim::Vec<3> force_pos;
    static int count_f = 300;
    exforce[0] = 0.;
    exforce[1] = 0;
    exforce[2] = 0;
    force_pos[0] = 0;
    force_pos[1] = 0;
    force_pos[2] = 0.33;
    count_f++;
    if(count_f%15000 < 120)
    {
        exforce[1] = 1400;
        panda5->setExternalForce (dj_id,panda5->BODY_FRAME,exforce,panda5->BODY_FRAME,force_pos);
    }
    // datafile << ceil(sqrt(exforce[1]*exforce[1]+exforce[0]*exforce[0]+exforce[2]*exforce[2]))<<",";

    // if(count_f%3020 < 10)
    //     panda5->setExternalForce (dj_id,panda5->BODY_FRAME,exforce,panda5->BODY_FRAME,force_pos);
    // if(count_f%3040 < 10)
    //     panda5->setExternalForce (dj_id,panda5->BODY_FRAME,exforce,panda5->BODY_FRAME,force_pos);
}
void sim_io(float dt)
{
    //sim to robot  sensor
    for(int i = 0;i<4;i++)
    {
        _sharedMemory().simToRobot.spiData.q_abad[i] = gc(0 + 3*i);
        _sharedMemory().simToRobot.spiData.q_hip[i] = gc(1 + 3*i);
        _sharedMemory().simToRobot.spiData.q_knee[i] = gc(2 + 3*i);

        _sharedMemory().simToRobot.spiData.qd_abad[i] = gv(0 + 3*i);
        _sharedMemory().simToRobot.spiData.qd_hip[i] = gv(1 + 3*i);
        _sharedMemory().simToRobot.spiData.qd_knee[i] = gv(2 + 3*i);
    }

    for(int i=0;i<6;i++)
    {
        _sharedMemory().simToRobot.spiData.q_arm[i] = gc(12+i);
        _sharedMemory().simToRobot.spiData.qd_arm[i] = gv(12+i);
    }

    float target_tau1_new[12];
    for (int leg = 0; leg < 4; leg++) {
      for (int joint = 0; joint < 3; joint++) {
//        target_tau1_new[leg * 3 + joint] = _actuatorModels[joint].getTorque(
//            tau_now[leg * 3 + joint],
//            gv[leg * 3 + joint]);
          target_tau1_new[leg * 3 + joint] = tau_now[leg * 3 + joint];
      }
    }
    for(int i = 0;i<4;i++)
    {
        _sharedMemory().simToRobot.spiData.tau_abad[i] = target_tau1_new[0 + 3*i];
        _sharedMemory().simToRobot.spiData.tau_hip[i] = target_tau1_new[1 + 3*i];
        _sharedMemory().simToRobot.spiData.tau_knee[i] = target_tau1_new[2 + 3*i];
    }

    FK(gc,tau_now);
    //IK();
    raisim::Vec<3> body_frame_p;
    raisim::Mat<3, 3> body_frame_orient;
    raisim::Vec<3> body_frame_v;
    raisim::Vec<3> body_frame_omega;
    Mat3<float> rt;
    panda5->getFramePosition(panda5->getFrameIdxByName("imu_joint"), body_frame_p);
    panda5->getFrameVelocity(panda5->getFrameIdxByName("imu_joint"), body_frame_v);
    panda5->getFrameOrientation(panda5->getFrameIdxByName("imu_joint"), body_frame_orient);
    panda5->getFrameAngularVelocity(panda5->getFrameIdxByName("imu_joint"), body_frame_omega);
    raisim::Vec<3> acc;
    Eigen::VectorXd force = panda5->getGeneralizedForce().e().head(3);
    Eigen::VectorXd force_all = panda5->getGeneralizedForce().e();
    // double mass = panda5->getTotalMass();
    raisim::Vec<4> quat;

    raisim::rotMatToQuat(body_frame_orient,quat);//ori::rotationMatrixToQuaternion(rt);
    Vec4<float>orientation_body = Vec4<float>(quat(0),quat(1),quat(2),quat(3));
    _sharedMemory().simToRobot.orientation = Vec4<float>(quat(0),quat(1),quat(2),quat(3));
    Vec4<float> bodyOrientation = _sharedMemory().simToRobot.orientation;
    Mat3<float> R_body = ori::quaternionToRotationMatrix(bodyOrientation).transpose();//ori::rpyToRotMat(ori::quatToRPY(bodyOrientation)).transpose();//ori::quaternionToRotationMatrix(bodyOrientation);//R_body *
    _sharedMemory().simToRobot.pos = Vec3<float>(body_frame_p(0),body_frame_p(1),body_frame_p(2));
    _sharedMemory().simToRobot.vel = Vec3<float>(body_frame_v(0),body_frame_v(1),body_frame_v(2));
    _sharedMemory().simToRobot.omega = R_body.transpose() * Vec3<float>(body_frame_omega(0),body_frame_omega(1),body_frame_omega(2));
    // printf("eluer %f %f %f %f \n",quat(0),quat(1),quat(2),quat(3));
    Vec3<float> foot_force_world[4];

    for(int i = 0;i<4;i++)
    {
        foot_force_world[i] = R_body * Vec3<float>(foot_force[i][0],foot_force[i][1],foot_force[i][2]);
        for(int j=0;j<3;j++)
        {
            _sharedMemory().simToRobot.spiData.foot_force[i][j] = foot_force_world[i][j];
        }
        // cout<<"force :"<<_sharedMemory().simToRobot.spiData.foot_force[i][2]<<endl;
    }
    Vec3<float> omega_body = _sharedMemory().simToRobot.omega;//omega_body
    Vec3<float> vel_body = R_body.transpose() *_sharedMemory().simToRobot.vel;//v_body
    SVec<float> vel_e;//
    vel_e(0) = omega_body(0);//,body_frame_omega(1),body_frame_omega(2),body_frame_v(0),body_frame_v(1),body_frame_v(2));
    vel_e(1) = omega_body(1);
    vel_e(2) = omega_body(2);
    vel_e(3) = vel_body(0);
    vel_e(4) = vel_body(1);
    vel_e(5) = vel_body(2);
    SVec<float> d_vel = (vel_e - last_vel)/dt;
    last_vel = vel_e;
    Vec3<float> acceleration = (R_body.transpose() * (Vec3<float>(0, 0, 9.81)) +
                         spatial::spatialToLinearAcceleration(
                             d_vel, vel_e));
    acc(0) = acceleration(0);
    acc(1) = acceleration(1);
    acc(2) = acceleration(2);
    _sharedMemory().simToRobot.acc = acceleration;//acc_body
    Eigen::VectorXd jointForwardTorque(dof_root + dof_leg + dof_arm); // 26(include root)
    jointForwardTorque.setZero();
    //joint PD controller
    Eigen::VectorXd jointNominalConfig1(dof_root + dof_leg + dof_arm+1), jointVelocityTarget1(dof_root + dof_leg + dof_arm);
    Eigen::VectorXd jointState1(dof_root + dof_leg + dof_arm), jointForce1(dof_root + dof_leg + dof_arm), jointPgain1(dof_root + dof_leg + dof_arm),
        jointDgain1(dof_root + dof_leg + dof_arm), ZERO(dof_root + dof_leg + dof_arm);
    raisim::VecDyn jointForceFeed(dof_root + dof_leg + dof_arm);
    ZERO.setZero();
    jointPgain1.setZero();
    jointDgain1.setZero();
    jointVelocityTarget1.setZero();
    jointNominalConfig1.setZero();
    jointVelocityTarget1.setZero();
    jointForce1.setZero();
    jointPgain1.setZero();
    jointDgain1.setZero();
    jointPgain1.tail(dof_root + dof_leg + dof_arm-6).setConstant(200.0);
    jointDgain1.tail(dof_root + dof_leg + dof_arm-6).setConstant(10.0);
#if _ARX_ARM_EN
    float target_q1[20] = {_sharedMemory().robotToSim.spiCommand.q_des_abad[0],
                           _sharedMemory().robotToSim.spiCommand.q_des_hip[0],
                           _sharedMemory().robotToSim.spiCommand.q_des_knee[0],
                           _sharedMemory().robotToSim.spiCommand.q_des_abad[1],
                           _sharedMemory().robotToSim.spiCommand.q_des_hip[1],
                           _sharedMemory().robotToSim.spiCommand.q_des_knee[1],
                           _sharedMemory().robotToSim.spiCommand.q_des_abad[2],
                           _sharedMemory().robotToSim.spiCommand.q_des_hip[2],
                           _sharedMemory().robotToSim.spiCommand.q_des_knee[2],
                           _sharedMemory().robotToSim.spiCommand.q_des_abad[3],
                           _sharedMemory().robotToSim.spiCommand.q_des_hip[3],
                           _sharedMemory().robotToSim.spiCommand.q_des_knee[3],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[0],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[1],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[2],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[3],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[4],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[5],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[6],
                           _sharedMemory().robotToSim.spiCommand.q_des_arm[7]};//desire input

    float target_qd1[20] = {_sharedMemory().robotToSim.spiCommand.qd_des_abad[0],
                            _sharedMemory().robotToSim.spiCommand.qd_des_hip[0],
                            _sharedMemory().robotToSim.spiCommand.qd_des_knee[0],
                            _sharedMemory().robotToSim.spiCommand.qd_des_abad[1],
                            _sharedMemory().robotToSim.spiCommand.qd_des_hip[1],
                            _sharedMemory().robotToSim.spiCommand.qd_des_knee[1],
                            _sharedMemory().robotToSim.spiCommand.qd_des_abad[2],
                            _sharedMemory().robotToSim.spiCommand.qd_des_hip[2],
                            _sharedMemory().robotToSim.spiCommand.qd_des_knee[2],
                            _sharedMemory().robotToSim.spiCommand.qd_des_abad[3],
                            _sharedMemory().robotToSim.spiCommand.qd_des_hip[3],
                            _sharedMemory().robotToSim.spiCommand.qd_des_knee[3],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[0],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[1],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[2],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[3],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[4],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[5],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[6],
                            _sharedMemory().robotToSim.spiCommand.qd_des_arm[7]};

    float target_tau1[20] = {_sharedMemory().robotToSim.spiCommand.tau_abad_ff[0],
                             _sharedMemory().robotToSim.spiCommand.tau_hip_ff[0],
                             _sharedMemory().robotToSim.spiCommand.tau_knee_ff[0],
                             _sharedMemory().robotToSim.spiCommand.tau_abad_ff[1],
                             _sharedMemory().robotToSim.spiCommand.tau_hip_ff[1],
                             _sharedMemory().robotToSim.spiCommand.tau_knee_ff[1],
                             _sharedMemory().robotToSim.spiCommand.tau_abad_ff[2],
                             _sharedMemory().robotToSim.spiCommand.tau_hip_ff[2],
                             _sharedMemory().robotToSim.spiCommand.tau_knee_ff[2],
                             _sharedMemory().robotToSim.spiCommand.tau_abad_ff[3],
                             _sharedMemory().robotToSim.spiCommand.tau_hip_ff[3],
                             _sharedMemory().robotToSim.spiCommand.tau_knee_ff[3],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[0],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[1],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[2],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[3],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[4],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[5],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[6],
                             _sharedMemory().robotToSim.spiCommand.tau_arm[7]};

    float kp_gain1[20] = {_sharedMemory().robotToSim.spiCommand.kp_abad[0],
                          _sharedMemory().robotToSim.spiCommand.kp_hip[0],
                          _sharedMemory().robotToSim.spiCommand.kp_knee[0],
                          _sharedMemory().robotToSim.spiCommand.kp_abad[1],
                          _sharedMemory().robotToSim.spiCommand.kp_hip[1],
                          _sharedMemory().robotToSim.spiCommand.kp_knee[1],
                          _sharedMemory().robotToSim.spiCommand.kp_abad[2],
                          _sharedMemory().robotToSim.spiCommand.kp_hip[2],
                          _sharedMemory().robotToSim.spiCommand.kp_knee[2],
                          _sharedMemory().robotToSim.spiCommand.kp_abad[3],
                          _sharedMemory().robotToSim.spiCommand.kp_hip[3],
                          _sharedMemory().robotToSim.spiCommand.kp_knee[3],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[0],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[1],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[2],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[3],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[4],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[5],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[6],
                          _sharedMemory().robotToSim.spiCommand.kp_arm[7]};

    float kd_gain1[20] = {_sharedMemory().robotToSim.spiCommand.kd_abad[0],
                          _sharedMemory().robotToSim.spiCommand.kd_hip[0],
                          _sharedMemory().robotToSim.spiCommand.kd_knee[0],
                          _sharedMemory().robotToSim.spiCommand.kd_abad[1],
                          _sharedMemory().robotToSim.spiCommand.kd_hip[1],
                          _sharedMemory().robotToSim.spiCommand.kd_knee[1],
                          _sharedMemory().robotToSim.spiCommand.kd_abad[2],
                          _sharedMemory().robotToSim.spiCommand.kd_hip[2],
                          _sharedMemory().robotToSim.spiCommand.kd_knee[2],
                          _sharedMemory().robotToSim.spiCommand.kd_abad[3],
                          _sharedMemory().robotToSim.spiCommand.kd_hip[3],
                          _sharedMemory().robotToSim.spiCommand.kd_knee[3],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[0],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[1],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[2],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[3],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[4],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[5],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[6],
                          _sharedMemory().robotToSim.spiCommand.kd_arm[7]};
#else
    float target_q1[12] = {_sharedMemory().robotToSim.spiCommand.q_des_abad[0],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_hip[0],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_knee[0],
                          _sharedMemory().robotToSim.spiCommand.q_des_abad[1],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_hip[1],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_knee[1],
                          _sharedMemory().robotToSim.spiCommand.q_des_abad[2],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_hip[2],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_knee[2],
                          _sharedMemory().robotToSim.spiCommand.q_des_abad[3],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_hip[3],
                                                   _sharedMemory().robotToSim.spiCommand.q_des_knee[3]};//desire input

    float target_qd1[12] = {_sharedMemory().robotToSim.spiCommand.qd_des_abad[0],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_hip[0],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_knee[0],
                          _sharedMemory().robotToSim.spiCommand.qd_des_abad[1],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_hip[1],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_knee[1],
                          _sharedMemory().robotToSim.spiCommand.qd_des_abad[2],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_hip[2],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_knee[2],
                          _sharedMemory().robotToSim.spiCommand.qd_des_abad[3],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_hip[3],
                                                   _sharedMemory().robotToSim.spiCommand.qd_des_knee[3]};


    float target_tau1[12] = {_sharedMemory().robotToSim.spiCommand.tau_abad_ff[0],
                                                   _sharedMemory().robotToSim.spiCommand.tau_hip_ff[0],
                                                   _sharedMemory().robotToSim.spiCommand.tau_knee_ff[0],
                          _sharedMemory().robotToSim.spiCommand.tau_abad_ff[1],
                                                   _sharedMemory().robotToSim.spiCommand.tau_hip_ff[1],
                                                   _sharedMemory().robotToSim.spiCommand.tau_knee_ff[1],
                          _sharedMemory().robotToSim.spiCommand.tau_abad_ff[2],
                                                   _sharedMemory().robotToSim.spiCommand.tau_hip_ff[2],
                                                   _sharedMemory().robotToSim.spiCommand.tau_knee_ff[2],
                          _sharedMemory().robotToSim.spiCommand.tau_abad_ff[3],
                                                   _sharedMemory().robotToSim.spiCommand.tau_hip_ff[3],
                                                   _sharedMemory().robotToSim.spiCommand.tau_knee_ff[3]};

    float kp_gain1[12] = {_sharedMemory().robotToSim.spiCommand.kp_abad[0],
                                                   _sharedMemory().robotToSim.spiCommand.kp_hip[0],
                                                   _sharedMemory().robotToSim.spiCommand.kp_knee[0],
                          _sharedMemory().robotToSim.spiCommand.kp_abad[1],
                                                   _sharedMemory().robotToSim.spiCommand.kp_hip[1],
                                                   _sharedMemory().robotToSim.spiCommand.kp_knee[1],
                          _sharedMemory().robotToSim.spiCommand.kp_abad[2],
                                                   _sharedMemory().robotToSim.spiCommand.kp_hip[2],
                                                   _sharedMemory().robotToSim.spiCommand.kp_knee[2],
                          _sharedMemory().robotToSim.spiCommand.kp_abad[3],
                                                   _sharedMemory().robotToSim.spiCommand.kp_hip[3],
                                                   _sharedMemory().robotToSim.spiCommand.kp_knee[3]};

    float kd_gain1[12] = {_sharedMemory().robotToSim.spiCommand.kd_abad[0],
                                                   _sharedMemory().robotToSim.spiCommand.kd_hip[0],
                                                   _sharedMemory().robotToSim.spiCommand.kd_knee[0],
                          _sharedMemory().robotToSim.spiCommand.kd_abad[1],
                                                   _sharedMemory().robotToSim.spiCommand.kd_hip[1],
                                                   _sharedMemory().robotToSim.spiCommand.kd_knee[1],
                          _sharedMemory().robotToSim.spiCommand.kd_abad[2],
                                                   _sharedMemory().robotToSim.spiCommand.kd_hip[2],
                                                   _sharedMemory().robotToSim.spiCommand.kd_knee[2],
                          _sharedMemory().robotToSim.spiCommand.kd_abad[3],
                                                   _sharedMemory().robotToSim.spiCommand.kd_hip[3],
                                                   _sharedMemory().robotToSim.spiCommand.kd_knee[3]};
#endif
        // printf("kp %f \n",kp_gain1[12]);
    if(1)
    {

        jointNominalConfig1.head(dof_root+1 + dof_leg + dof_arm)<<0, 0, 0.2, 1, 0.0, 0, 0.0,
            target_q1[0],target_q1[1],target_q1[2],
                   target_q1[3],target_q1[4],target_q1[5],
                    target_q1[6],target_q1[7],target_q1[8],
                    target_q1[9],target_q1[10],target_q1[11],
            target_q1[12],target_q1[13],target_q1[14],
            target_q1[15],target_q1[16],target_q1[17],
            target_q1[18],target_q1[19];

        jointVelocityTarget1.head(dof_root + dof_leg + dof_arm).tail(20)<<target_qd1[0],target_qd1[1],target_qd1[2],
                target_qd1[3],target_qd1[4],target_qd1[5],
                target_qd1[6],target_qd1[7],target_qd1[8],
                target_qd1[9],target_qd1[10],target_qd1[11],
            target_qd1[12],target_qd1[13],target_qd1[14],
            target_qd1[15],target_qd1[16],target_qd1[17],
            target_qd1[18],target_qd1[19];

        jointForce1.head(dof_root + dof_leg + dof_arm).tail(20)<<target_tau1[0],target_tau1[1],target_tau1[2],
                target_tau1[3],target_tau1[4],target_tau1[5],
                target_tau1[6],target_tau1[7],target_tau1[8],
                target_tau1[9],target_tau1[10],target_tau1[11],
            target_tau1[12],target_tau1[13],target_tau1[14],
            target_tau1[15],target_tau1[16],target_tau1[17],
            target_tau1[18],target_tau1[19];

        jointPgain1.head(dof_root + dof_leg + dof_arm).tail(20)<<kp_gain1[0],kp_gain1[1],kp_gain1[2],
                kp_gain1[3],kp_gain1[4],kp_gain1[5],
                kp_gain1[6],kp_gain1[7],kp_gain1[8],
                kp_gain1[9],kp_gain1[10],kp_gain1[11],
            kp_gain1[12],kp_gain1[13],kp_gain1[14],
            kp_gain1[15],kp_gain1[16],kp_gain1[17],
            kp_gain1[18],kp_gain1[19];

        jointDgain1.head(dof_root + dof_leg + dof_arm).tail(20)<<kd_gain1[0],kd_gain1[1],kd_gain1[2],
                kd_gain1[3],kd_gain1[4],kd_gain1[5],
                kd_gain1[6],kd_gain1[7],kd_gain1[8],
                kd_gain1[9],kd_gain1[10],kd_gain1[11],
            kd_gain1[12],kd_gain1[13],kd_gain1[14],
            kd_gain1[15],kd_gain1[16],kd_gain1[17],
            kd_gain1[18],kd_gain1[19];
    }
    else {
            jointNominalConfig1 << 0, 0, 0.2, 1, 0, 0, 0.0, 0.0, 0.3, -1, 0.0, 1, -2.3,
            0.0, 0.3, -1, 0.0, 1, -2.3,0,0,0,0,0,0,0;
    }
    panda5->setPdGains(jointPgain1, jointDgain1);
    // cout<<jointForce1<<endl;
    // cout<<""<<endl;
    panda5->setGeneralizedForce(jointForce1);

    panda5->setPdTarget(jointNominalConfig1, jointVelocityTarget1);



    // /*-------------------- start raisim simulation target---------------------- */

    // // 22222 is calculated by PD controller

    // // 前馈扭矩: jointForwardTorque前6个为广义坐标，
    // // jointNominalConfig2 is qd, jointVelocityTarget2 is dqd

    // for (int wj = 0; wj < 18; wj++)
    // {
    //     jointForwardTorque(6 + wj) = jointPgain1(6 + wj) * (jointNominalConfig1(7 + wj) - gc(wj)) +
    //                                  jointDgain1(6 + wj) * (jointVelocityTarget1(6 + wj) - gv(wj));

    //     // if (wj == 0)
    //     // {
    //     //     cout << "jointForwardTorque:" << jointForwardTorque(6 + wj) << endl;
    //     //     cout << "jointPgain1:" << jointPgain1(wj) << endl;
    //     //     cout << "jointNominalConfig" << jointNominalConfig1(7 + wj) << endl;
    //     //     cout << "gc:" << gc(wj) << endl;
    //     //     cout << "jointDgain1:" << jointDgain1(wj) << endl;
    //     //     cout << "jointVelocityTarget:" << jointVelocityTarget1(6 + wj) << endl;
    //     //     cout << "gv:" << gv(wj) << endl;
    //     // }
    // }

    // // cout << "P:" << jointPgain1.transpose() << endl;
    // // cout << "D:" << jointDgain1.transpose() << endl;
    // // cout << "jointNominalConfig1:" << jointNominalConfig1.transpose() << endl;
    // // cout << "jointVelocityTarget1:" << jointVelocityTarget1.transpose() << endl;

    // // cout << "jointForwardTorque:" << jointForwardTorque.transpose() << endl;
    // // cout << "gc:" << gc.transpose() << endl;
    // // cout << "gv:" << gv.transpose() << endl;

    // // 反馈扭矩设为0
    // panda5->setPdGains(ZERO, ZERO);

    // panda5->setGeneralizedForce(jointForwardTorque);

    // panda5->setPdTarget(jointNominalConfig1, jointVelocityTarget1);

}


void sim_io2(float dt)
{
    //sim to robot  sensor

    for(int i = 0;i<4;i++)
    {
        _sharedMemory2().simToRobot.spiData.q_abad[i] = gc2(0 + 3*i);
        _sharedMemory2().simToRobot.spiData.q_hip[i] = gc2(1 + 3*i);
        _sharedMemory2().simToRobot.spiData.q_knee[i] = gc2(2 + 3*i);

        _sharedMemory2().simToRobot.spiData.qd_abad[i] = gv2(0 + 3*i);
        _sharedMemory2().simToRobot.spiData.qd_hip[i] = gv2(1 + 3*i);
        _sharedMemory2().simToRobot.spiData.qd_knee[i] = gv2(2 + 3*i);
    }
    for(int i=0;i<6;i++)
    {
        _sharedMemory2().simToRobot.spiData.q_arm[i] = gc2(12+i);
        _sharedMemory2().simToRobot.spiData.qd_arm[i] = gv2(12+i);
    }

    float target_tau1_new[12];
    for (int leg = 0; leg < 4; leg++) {
        for (int joint = 0; joint < 3; joint++) {
            //        target_tau1_new[leg * 3 + joint] = _actuatorModels[joint].getTorque(
            //            tau_now[leg * 3 + joint],
            //            gv[leg * 3 + joint]);
            target_tau1_new[leg * 3 + joint] = tau_now2[leg * 3 + joint];
        }
    }
    for(int i = 0;i<4;i++)
    {
        _sharedMemory2().simToRobot.spiData.tau_abad[i] = target_tau1_new[0 + 3*i];
        _sharedMemory2().simToRobot.spiData.tau_hip[i] = target_tau1_new[1 + 3*i];
        _sharedMemory2().simToRobot.spiData.tau_knee[i] = target_tau1_new[2 + 3*i];
    }

    FK(gc2,tau_now2);
    //IK();
    raisim::Vec<3> body_frame_p;
    raisim::Mat<3, 3> body_frame_orient;
    raisim::Vec<3> body_frame_v;
    raisim::Vec<3> body_frame_omega;
    Mat3<float> rt;
    // std::cout<<"frame id : "<<panda7->getFrameIdxByName("imu_joint")<<std::endl;
    panda7->getFramePosition(panda7->getFrameIdxByName("imu_joint"), body_frame_p);
    panda7->getFrameVelocity(panda7->getFrameIdxByName("imu_joint"), body_frame_v);
    panda7->getFrameOrientation(panda7->getFrameIdxByName("imu_joint"), body_frame_orient);
    panda7->getFrameAngularVelocity(panda7->getFrameIdxByName("imu_joint"), body_frame_omega);
    raisim::Vec<3> acc;
    Eigen::VectorXd force = panda7->getGeneralizedForce().e().head(3);
    Eigen::VectorXd force_all = panda7->getGeneralizedForce().e();
    double mass = panda7->getTotalMass();
    raisim::Vec<4> quat;
    raisim::rotMatToQuat(body_frame_orient,quat);//ori::rotationMatrixToQuaternion(rt);
    Vec4<float>orientation_body = Vec4<float>(quat(0),quat(1),quat(2),quat(3));
    _sharedMemory2().simToRobot.orientation = Vec4<float>(quat(0),quat(1),quat(2),quat(3));
    Vec4<float> bodyOrientation = _sharedMemory2().simToRobot.orientation;
    Mat3<float> R_body = ori::quaternionToRotationMatrix(bodyOrientation).transpose();//ori::rpyToRotMat(ori::quatToRPY(bodyOrientation)).transpose();//ori::quaternionToRotationMatrix(bodyOrientation);//R_body *
    _sharedMemory2().simToRobot.pos = Vec3<float>(body_frame_p(0),body_frame_p(1),body_frame_p(2));
    _sharedMemory2().simToRobot.vel = Vec3<float>(body_frame_v(0),body_frame_v(1),body_frame_v(2));
    _sharedMemory2().simToRobot.omega = R_body.transpose() * Vec3<float>(body_frame_omega(0),body_frame_omega(1),body_frame_omega(2));
    // printf("eluer %f %f %f \n",body_frame_omega(0),body_frame_omega(1),body_frame_omega(2));
    Vec3<float> foot_force_world[4];
    for(int i = 0;i<4;i++)
    {
        foot_force_world[i] = R_body * Vec3<float>(foot_force2[i][0],foot_force2[i][1],foot_force2[i][2]);
        for(int j=0;j<3;j++)
        {
            _sharedMemory2().simToRobot.spiData.foot_force[i][j] = foot_force_world[i][j];
        }
        // cout<<"force :"<<_sharedMemory().simToRobot.spiData.foot_force[i][2]<<endl;
    }
    Vec3<float> omega_body = _sharedMemory2().simToRobot.omega;//omega_body
    Vec3<float> vel_body = R_body.transpose() *_sharedMemory2().simToRobot.vel;//v_body
    SVec<float> vel_e;//
    vel_e(0) = omega_body(0);//,body_frame_omega(1),body_frame_omega(2),body_frame_v(0),body_frame_v(1),body_frame_v(2));
    vel_e(1) = omega_body(1);
    vel_e(2) = omega_body(2);
    vel_e(3) = vel_body(0);
    vel_e(4) = vel_body(1);
    vel_e(5) = vel_body(2);
    SVec<float> d_vel = (vel_e - last_vel2)/dt;
    last_vel2 = vel_e;
    Vec3<float> acceleration = (R_body.transpose() * (Vec3<float>(0, 0, 9.81)) +
                                spatial::spatialToLinearAcceleration(
                                    d_vel, vel_e));
    acc(0) = acceleration(0);
    acc(1) = acceleration(1);
    acc(2) = acceleration(2);
    _sharedMemory2().simToRobot.acc = acceleration;//acc_body
    //joint PD controller
    Eigen::VectorXd jointNominalConfig2(dof_root + dof_leg + dof_arm+1), jointVelocityTarget2(dof_root + dof_leg + dof_arm);
    Eigen::VectorXd jointState2(dof_root + dof_leg + dof_arm), jointForce2(dof_root + dof_leg + dof_arm), jointPgain2(dof_root + dof_leg + dof_arm),
        jointDgain2(dof_root + dof_leg + dof_arm);
    raisim::VecDyn jointForceFeed(dof_root + dof_leg + dof_arm);
    jointPgain2.setZero();
    jointDgain2.setZero();
    jointVelocityTarget2.setZero();
    jointNominalConfig2.setZero();
    jointVelocityTarget2.setZero();
    jointForce2.setZero();
    jointPgain2.setZero();
    jointDgain2.setZero();
    jointPgain2.tail(dof_root + dof_leg + dof_arm - 6).setConstant(200.0);
    jointDgain2.tail(dof_root + dof_leg + dof_arm - 6).setConstant(10.0);

    float target_q2[20] = {_sharedMemory2().robotToSim.spiCommand.q_des_abad[0],
                           _sharedMemory2().robotToSim.spiCommand.q_des_hip[0],
                           _sharedMemory2().robotToSim.spiCommand.q_des_knee[0],
                           _sharedMemory2().robotToSim.spiCommand.q_des_abad[1],
                           _sharedMemory2().robotToSim.spiCommand.q_des_hip[1],
                           _sharedMemory2().robotToSim.spiCommand.q_des_knee[1],
                           _sharedMemory2().robotToSim.spiCommand.q_des_abad[2],
                           _sharedMemory2().robotToSim.spiCommand.q_des_hip[2],
                           _sharedMemory2().robotToSim.spiCommand.q_des_knee[2],
                           _sharedMemory2().robotToSim.spiCommand.q_des_abad[3],
                           _sharedMemory2().robotToSim.spiCommand.q_des_hip[3],
                           _sharedMemory2().robotToSim.spiCommand.q_des_knee[3],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[0],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[1],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[2],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[3],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[4],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[5],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[6],
                           _sharedMemory2().robotToSim.spiCommand.q_des_arm[7]};//desire input

    float target_qd2[20] = {_sharedMemory2().robotToSim.spiCommand.qd_des_abad[0],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_hip[0],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_knee[0],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_abad[1],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_hip[1],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_knee[1],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_abad[2],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_hip[2],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_knee[2],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_abad[3],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_hip[3],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_knee[3],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[0],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[1],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[2],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[3],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[4],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[5],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[6],
                            _sharedMemory2().robotToSim.spiCommand.qd_des_arm[7]};

    float target_tau2[20] = {_sharedMemory2().robotToSim.spiCommand.tau_abad_ff[0],
                             _sharedMemory2().robotToSim.spiCommand.tau_hip_ff[0],
                             _sharedMemory2().robotToSim.spiCommand.tau_knee_ff[0],
                             _sharedMemory2().robotToSim.spiCommand.tau_abad_ff[1],
                             _sharedMemory2().robotToSim.spiCommand.tau_hip_ff[1],
                             _sharedMemory2().robotToSim.spiCommand.tau_knee_ff[1],
                             _sharedMemory2().robotToSim.spiCommand.tau_abad_ff[2],
                             _sharedMemory2().robotToSim.spiCommand.tau_hip_ff[2],
                             _sharedMemory2().robotToSim.spiCommand.tau_knee_ff[2],
                             _sharedMemory2().robotToSim.spiCommand.tau_abad_ff[3],
                             _sharedMemory2().robotToSim.spiCommand.tau_hip_ff[3],
                             _sharedMemory2().robotToSim.spiCommand.tau_knee_ff[3],
                             _sharedMemory2().robotToSim.spiCommand.tau_arm[0],
                             _sharedMemory2().robotToSim.spiCommand.tau_arm[1],
                             _sharedMemory2().robotToSim.spiCommand.tau_arm[2],
                             _sharedMemory2().robotToSim.spiCommand.tau_arm[3],
                             _sharedMemory2().robotToSim.spiCommand.tau_arm[4],
                             _sharedMemory2().robotToSim.spiCommand.tau_arm[5],
                              _sharedMemory2().robotToSim.spiCommand.tau_arm[6],
                              _sharedMemory2().robotToSim.spiCommand.tau_arm[7]};

    float kp_gain2[20] = {_sharedMemory2().robotToSim.spiCommand.kp_abad[0],
                          _sharedMemory2().robotToSim.spiCommand.kp_hip[0],
                          _sharedMemory2().robotToSim.spiCommand.kp_knee[0],
                          _sharedMemory2().robotToSim.spiCommand.kp_abad[1],
                          _sharedMemory2().robotToSim.spiCommand.kp_hip[1],
                          _sharedMemory2().robotToSim.spiCommand.kp_knee[1],
                          _sharedMemory2().robotToSim.spiCommand.kp_abad[2],
                          _sharedMemory2().robotToSim.spiCommand.kp_hip[2],
                          _sharedMemory2().robotToSim.spiCommand.kp_knee[2],
                          _sharedMemory2().robotToSim.spiCommand.kp_abad[3],
                          _sharedMemory2().robotToSim.spiCommand.kp_hip[3],
                          _sharedMemory2().robotToSim.spiCommand.kp_knee[3],
                          _sharedMemory2().robotToSim.spiCommand.kp_arm[0],
                          _sharedMemory2().robotToSim.spiCommand.kp_arm[1],
                          _sharedMemory2().robotToSim.spiCommand.kp_arm[2],
                          _sharedMemory2().robotToSim.spiCommand.kp_arm[3],
                          _sharedMemory2().robotToSim.spiCommand.kp_arm[4],
                          _sharedMemory2().robotToSim.spiCommand.kp_arm[5],
                           _sharedMemory2().robotToSim.spiCommand.kp_arm[6],
                           _sharedMemory2().robotToSim.spiCommand.kp_arm[7]};

    float kd_gain2[20] = {_sharedMemory2().robotToSim.spiCommand.kd_abad[0],
                          _sharedMemory2().robotToSim.spiCommand.kd_hip[0],
                          _sharedMemory2().robotToSim.spiCommand.kd_knee[0],
                          _sharedMemory2().robotToSim.spiCommand.kd_abad[1],
                          _sharedMemory2().robotToSim.spiCommand.kd_hip[1],
                          _sharedMemory2().robotToSim.spiCommand.kd_knee[1],
                          _sharedMemory2().robotToSim.spiCommand.kd_abad[2],
                          _sharedMemory2().robotToSim.spiCommand.kd_hip[2],
                          _sharedMemory2().robotToSim.spiCommand.kd_knee[2],
                          _sharedMemory2().robotToSim.spiCommand.kd_abad[3],
                          _sharedMemory2().robotToSim.spiCommand.kd_hip[3],
                          _sharedMemory2().robotToSim.spiCommand.kd_knee[3],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[0],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[1],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[2],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[3],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[4],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[5],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[6],
                          _sharedMemory2().robotToSim.spiCommand.kd_arm[7]};

    if(1)
    {
        jointNominalConfig2.head(dof_root+1 + dof_leg + dof_arm)<<0, 0, 0.2, 1, 0.0, 0.0, 0.0,
            target_q2[0],target_q2[1],target_q2[2],
            target_q2[3],target_q2[4],target_q2[5],
            target_q2[6],target_q2[7],target_q2[8],
            target_q2[9],target_q2[10],target_q2[11],
            target_q2[12],target_q2[13],target_q2[14],
            target_q2[15],target_q2[16],target_q2[17],
            target_q2[18],target_q2[19];


        jointVelocityTarget2.head(dof_root + dof_leg + dof_arm).tail(20)<<target_qd2[0],target_qd2[1],target_qd2[2],
            target_qd2[3],target_qd2[4],target_qd2[5],
            target_qd2[6],target_qd2[7],target_qd2[8],
            target_qd2[9],target_qd2[10],target_qd2[11],
            target_qd2[12],target_qd2[13],target_qd2[14],
            target_qd2[15],target_qd2[16],target_qd2[17],
            target_qd2[18],target_qd2[19];

        jointForce2.head(dof_root + dof_leg + dof_arm).tail(20)<<target_tau2[0],target_tau2[1],target_tau2[2],
            target_tau2[3],target_tau2[4],target_tau2[5],
            target_tau2[6],target_tau2[7],target_tau2[8],
            target_tau2[9],target_tau2[10],target_tau2[11],
            target_tau2[12],target_tau2[13],target_tau2[14],
            target_tau2[15],target_tau2[16],target_tau2[17],
            target_tau2[18],target_tau2[19];

        jointPgain2.head(dof_root + dof_leg + dof_arm).tail(20)<<kp_gain2[0],kp_gain2[1],kp_gain2[2],
            kp_gain2[3],kp_gain2[4],kp_gain2[5],
            kp_gain2[6],kp_gain2[7],kp_gain2[8],
            kp_gain2[9],kp_gain2[10],kp_gain2[11],
            kp_gain2[12],kp_gain2[13],kp_gain2[14],
            kp_gain2[15],kp_gain2[16],kp_gain2[17],
            kp_gain2[18],kp_gain2[19];

        jointDgain2.head(dof_root + dof_leg + dof_arm).tail(20)<<kd_gain2[0],kd_gain2[1],kd_gain2[2],
            kd_gain2[3],kd_gain2[4],kd_gain2[5],
            kd_gain2[6],kd_gain2[7],kd_gain2[8],
            kd_gain2[9],kd_gain2[10],kd_gain2[11],
            kd_gain2[12],kd_gain2[13],kd_gain2[14],
            kd_gain2[15],kd_gain2[16],kd_gain2[17],
            kd_gain2[18],kd_gain2[19];
    }
    else {
        jointNominalConfig2 << 0, 0, 0.2, 1, 0, 0, 0.0, 0.0, 0.3, -1, 0.0, 1, -2.3,
            0.0, 0.3, -1, 0.0, 1, -2.3,0,0,0,0,0,0,0,0;
    }
    // cout<<"P  "<<endl;
    // for (int i = 0; i < jointPgain2.size(); ++i) {
    //     std::cout << jointPgain2[i] << " ";  // 输出每个元素后跟一个空格
    // }
    // cout<<endl;
    // cout<<"D  "<<endl;
    // for (int i = 0; i < jointDgain2.size(); ++i) {
    //     std::cout << jointDgain2[i] << " ";  // 输出每个元素后跟一个空格
    // }
    // cout<<endl;
    // cout<<"F  "<<endl;
    // for (int i = 0; i < jointForce2.size(); ++i) {
    //     std::cout << jointForce2[i] << " ";  // 输出每个元素后跟一个空格
    // }
    // cout<<endl;
    // cout<<"V  "<<endl;
    // for (int i = 0; i < jointVelocityTarget2.size(); ++i) {
    //     std::cout << jointVelocityTarget2[i] << " ";  // 输出每个元素后跟一个空格
    // }
    // cout<<endl;
    // cout<<"jointNominalConfig2  "<<endl;
    // for (int i = 0; i < jointNominalConfig2.size(); ++i) {
    //     std::cout << jointNominalConfig2[i] << " ";  // 输出每个元素后跟一个空格
    // }
    // cout<<endl;
    // cout<<"jointVelocityTarget2  "<<endl;
    // for (int i = 0; i < jointVelocityTarget2.size(); ++i) {
    //     std::cout << jointVelocityTarget2[i] << " ";  // 输出每个元素后跟一个空格
    // }
    // cout<<endl;

    panda7->setPdGains(jointPgain2, jointDgain2);

    panda7->setGeneralizedForce(jointForce2);

    panda7->setPdTarget(jointNominalConfig2, jointVelocityTarget2);
}



int main(int argc, char* argv[]) {
    auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    /// create raisim world
    raisim::World world;

    world.setTimeStep(dt_i*1.0e-03);
    last_vel.setZero();
    auto ground = world.addGround();
    ground->setAppearance("dune");
    for(int i=0;i<4;i++)
    {
        world.setMaterialPairProp("default","foot"+std::to_string(i),0.7,0.0,0.0);
    }
    world.setDefaultMaterial(0.7,0.0,0.0);
    Eigen::VectorXd jointNominalConfig(num_dof+1), jointVelocityTarget(num_dof);//+1 quat
    Eigen::VectorXd jointNominalConfig2(num_dof+1), jointVelocityTarget2(num_dof);
    Eigen::VectorXd jointNominalConfig3(num_dof+1), jointVelocityTarget3(num_dof);
    Eigen::VectorXd jointState(num_dof), jointForce(num_dof), jointPgain(num_dof),jointDgain(num_dof);
    jointPgain.setZero();
    jointDgain.setZero();
    jointVelocityTarget.setZero();
    Vec3<float> rpy_d;
    rpy_d[0] = 0.0f;
    rpy_d[1] = 0.0f;
    rpy_d[2] = 0.0f;
    Quat<float> quat_d = ori::rpyToQuat(rpy_d);
    // 右前左前右后左后
    jointNominalConfig << 0, -1, 0.13, quat_d(0), quat_d(1), quat_d(2), quat_d(3), 0.1, 2.3, -2.4, -0.1, 2.3, -2.4,
        0.1, 2.3, -2.4, -0.1, 2.3, -2.4,0,0,0,0,0,0,0,0;//panda5
    jointNominalConfig2 <<0, 1, 0.13, quat_d(0), quat_d(1), quat_d(2), quat_d(3), 0.1, 2.3, -2.4, -0.1, 2.3, -2.4,
        0.1, 2.3, -2.4, -0.1, 2.3, -2.4,0,0,0,0,0,0,0,0;
    jointNominalConfig3 << 0, -1, 0.55, quat_d(0), quat_d(1), quat_d(2), quat_d(3), -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, 0, 0, 0, 0, 0, 0,0,0;//panda5
    jointPgain.tail(num_dof-6).setConstant(120.0);
    jointDgain.tail(num_dof-6).setConstant(1.0);

    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "/user/DogSimRaisim/rsc/_____asm.SLDASM/urdf/_____asm.SLDASM.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\panda\\panda7\\urdf\\panda7.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\panda7_oldleg_new\\urdf\\panda7.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim2\\rsc\\aubo\\urdf\\panda7.urdf");
    // panda7 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim2\\rsc\\aubo\\urdf\\panda7.urdf");

    panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim2\\rsc\\panda7_nleg_arm_1008\\panda7_nleg_arm_1227.urdf");
    panda7 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim2\\rsc\\panda7_nleg_arm_1008\\panda7_nleg_arm_1227.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\pandajm_dj_rp\\pandajm.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\panda7-oleg-724\\urdf\\panda7-oleg-724.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\panda7-oleg-728\\urdf\\panda7-oleg-728.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\aliengo_description\\urdf\\aliengo.urdf");//aliengo
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\panda\\panda5\\urdf\\panda5.urdf");//this
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\dl_00_00_jh_asm_r10_asm.SLDASM\\urdf\\dl_00_00_jh_asm_r10_asm.SLDASM.urdf");
    // panda5 = world.addArticulatedSystem(binaryPath.getDirectory() + "\\user\\DogSimRaisim\\rsc\\panda5\\urdf\\panda5.urdf");
    panda5->updateMassInfo();
    panda7->updateMassInfo();
    std::vector<double> mass = panda5->getMass();
    double sum_mass = 0, load_mass = 0;
    for(int i=0;i<mass.size();i++)
    {
        sum_mass+=mass.at(i);
        if(i>=12 && (_USE_TUO||_USE_TUO4))
            load_mass+=mass.at(i);
    }
    printf("Dof %d Sum_Mass %f Load_Mass %f  g %f\n",panda5->getDOF(),sum_mass,load_mass,world.getGravity()[2]);
    std::vector<raisim::Mat<3,3>> Inertia = panda5->getInertia();
    for (int i = 0; i < num_dof-5; ++i) {
        std::cout << "Joint " << i << ": " << panda7->getMovableJointNames()[i] << std::endl;  // 打印关节信息
    }
    panda7->setGeneralizedCoordinate(jointNominalConfig2);
    cout<<"doff "<<panda7->getDOF()<<endl;
    panda7->setGeneralizedForce(Eigen::VectorXd::Zero(panda7->getDOF()));
    panda7->setPdGains(jointPgain, jointDgain);
    panda7->setPdTarget(jointNominalConfig2, jointVelocityTarget2);
    panda7->setName("panda7");

    panda5->setGeneralizedCoordinate(jointNominalConfig);
    panda5->setGeneralizedForce(Eigen::VectorXd::Zero(panda5->getDOF()));
    panda5->setPdGains(jointPgain, jointDgain);
    panda5->setPdTarget(jointNominalConfig, jointVelocityTarget);
    panda5->setName("panda5");


    // launch raisim server
    raisim::RaisimServer server(&world);
    server.launchServer();
    _sharedMemory.createNew(DEVELOPMENT_SIMULATOR_SHARED_MEMORY_NAME, true);//zhy
    _sharedMemory().init();

    _sharedMemory2.createNew(DEVELOPMENT_SIMULATOR_SHARED_MEMORY_NAME2, true);//zhy
    _sharedMemory2().init();

    Eigen::VectorXd gc_i,gv_i,gc_i2,gv_i2;
    panda5->getState(gc_i,gv_i);
    panda7->getState(gc_i2,gv_i2);

    static double gc_last[12];
    gc = gc_i.tail(num_dof-6);
    gv = gv_i.tail(num_dof-6);
    pos = gc_i.head(3);
    orient = gv_i.head(7).tail(4);
    tau_now = panda5->getGeneralizedForce().e().tail(num_dof-6);


    gc2 = gc_i2.tail(num_dof-6);
    gv2 = gv_i2.tail(num_dof-6);
    pos2 = gc_i2.head(3);
    orient2 = gv_i2.head(7).tail(4);
    tau_now2 = panda7->getGeneralizedForce().e().tail(num_dof-6);

    for(int i = 0;i<4;i++)
    {
        _sharedMemory().simToRobot.spiData.q_abad[i] = gc(0 + 3*i);
        _sharedMemory().simToRobot.spiData.q_hip[i] = gc(1 + 3*i);
        _sharedMemory().simToRobot.spiData.q_knee[i] = gc(2 + 3*i);

        _sharedMemory().simToRobot.spiData.qd_abad[i] = gv(0 + 3*i);
        _sharedMemory().simToRobot.spiData.qd_hip[i] = gv(1 + 3*i);
        _sharedMemory().simToRobot.spiData.qd_knee[i] = gv(2 + 3*i);


        _sharedMemory2().simToRobot.spiData.q_abad[i] = gc2(0 + 3*i);
        _sharedMemory2().simToRobot.spiData.q_hip[i] = gc2(1 + 3*i);
        _sharedMemory2().simToRobot.spiData.q_knee[i] = gc2(2 + 3*i);

        _sharedMemory2().simToRobot.spiData.qd_abad[i] = gv2(0 + 3*i);
        _sharedMemory2().simToRobot.spiData.qd_hip[i] = gv2(1 + 3*i);
        _sharedMemory2().simToRobot.spiData.qd_knee[i] = gv2(2 + 3*i);
    }
    for(int i=0;i<7;i++)
    {
        _sharedMemory().simToRobot.spiData.q_arm[i] = gc(12+i);
        _sharedMemory2().simToRobot.spiData.q_arm[i] = gc2(12+i);

        _sharedMemory().simToRobot.spiData.qd_arm[i] = gv(12+i);
        _sharedMemory2().simToRobot.spiData.qd_arm[i] = gv2(12+i);
    }

    raisim::Vec<3> com_set_in,com_set_in2;
    com_set_in =  panda5->getBodyCOM_B()[0];
    com_set_in2 =  panda7->getBodyCOM_B()[0];

    long t = 0;
    static int sim_start = 0;
    static struct timespec time1 = {0, 0};
    static struct timespec time2 = {0, 0};
    static float timesim =0 ;
    float loop_time = (float)dt_i;//ms
    if(remove(datapath)==0)
        printf("delete success\n");
    datafile.open(datapath,ios::out);
    static Eigen::VectorXd gv_=gv;
    int counter = 0;

    // sleep(50);

  for ( ; ; ) {

    clock_gettime(CLOCK_REALTIME, &time1);
    //raisim::MSLEEP(dt_i);
    server.integrateWorldThreadSafe();
    Eigen::VectorXd gc_all,gv_all,gc_all2,gv_all2;
    //sample joint q qd tau
    panda5->getState(gc_all,gv_all);
    panda7->getState(gc_all2,gv_all2);

    double v ;
    v = sqrt(pow(gv_all[0],2)+pow(gv_all[1],2)+pow(gv_all[2],2));
    // std::cout<<"vel: "<<v<<endl;

    gc = gc_all.tail(20).head(18);//夹爪移动设的两个自由度，实际是一个自由度
    // cout<<"gc :"<<gc.transpose()<<endl;
    gv = gv_all.tail(20).head(18);//夹爪移动设的两个自由度，实际是一个自由度
    pos = gc_all.head(3);
    orient = gc_all.head(7).tail(4);

    gc2 = gc_all2.tail(20).head(18);
    gv2 = gv_all2.tail(20).head(18);
    pos2 = gc_all2.head(3);
    orient2 = gc_all2.head(7).tail(4);
    toEulerAngle(orient[1],orient[2],orient[3],orient[0],roll,pitch,yaw);
    toEulerAngle(orient2[1],orient2[2],orient2[3],orient2[0],roll2,pitch2,yaw2);
    Vec3<double> RPY,RPY2;
    RPY[0] = roll;
    RPY[1] = pitch;
    RPY[2] = yaw;
    RPY2[0] = roll2;
    RPY2[1] = pitch2;
    RPY2[2] = yaw2;
    tau_now = panda5->getGeneralizedForce().e().tail(num_dof-6);
    tau_arm = panda5->getGeneralizedForce().e().tail(8);

    // printf("tau leg %f %f %f %f %f %f \n",tau_now[6],tau_now[7],tau_now[8],tau_now[9],tau_now[10],tau_now[11]);

    // printf("tau arm %f %f %f %f %f %f \n",tau_now[12],tau_now[13],tau_now[14],tau_now[15],tau_now[16],tau_now[17]);

    // cout.precision(2);
    // cout<<"tau "<<tau_now.transpose()<<endl;

    tau_now2 = panda7->getGeneralizedForce().e().tail(num_dof-6);
    tau_arm2 = panda7->getGeneralizedForce().e().tail(8);


    // extern_force();

    // gv_ = gv*0.2+0.8*gv_;
    double pow_all = 0;
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            if(i==0)
                if(abs(gv[j*3+i])>19.3)
                    gv[j*3+i] = 19.3*(gv[j*3+i]/abs(gv[j*3+i]+0.00000000001));
            if(i==1)
                if(abs(gv[j*3+i])>21.6)
                    gv[j*3+i] = 21.6*(gv[j*3+i]/abs(gv[j*3+i]+0.00000000001));
            if(i==2)
                if(abs(gv[j*3+i])>12.8)
                    gv[j*3+i] = 12.8*(gv[j*3+i]/abs(gv[j*3+i]+0.00000000001));
            pow_all += abs(gv[j*3+i]*tau_now[j*3+i]);
        }
    }
    // datafile << pow_all;
    // datafile << "\n";

    if(pow_all > 4500)
        printf("######################over power: %f#####################\n",pow_all);

    raisim::Vec<3> com_set_new = com_set_in;
    panda5->getBodyCOM_B()[0] = com_set_new;
    panda5->updateMassInfo();

    raisim::Vec<3> com_set_new2 = com_set_in2;
    panda7->getBodyCOM_B()[0] = com_set_new2;
    panda7->updateMassInfo();

    if(t>=30000)
    {
        sim_start = 1;
        t = 30000;
    }

    // if(counter%4000==0)
    // {
    //     panda5->setGeneralizedCoordinate(jointNominalConfig3);
    //     printf("reset\n");
    // }

    // 添加外力干扰
    // if(counter%1==0)
    // {
    //     raisim::Vec<3> torque = {100,0,0};
    //     torque[0] = 100*sin(3.14*counter/5);
    //     panda5->setExternalTorqueInBodyFrame(panda5->getBodyIdx("base"), torque);
    //     cout<<"add torque"<<endl;
    // }
    // ++counter;
    
    if(sim_start==1)
    {

        sim_io(dt_i*1.0e-03);  //1.0e-03
        // sim_io2(dt_i*1.0e-03);
    }
    t+=dt_i;
    clock_gettime(CLOCK_REALTIME, &time2);
    timesim = (float)(time2.tv_sec - time1.tv_sec)*1000.0 + (float)(time2.tv_nsec - time1.tv_nsec)/1000000.0;
    if(timesim < loop_time)
    {
        usleep((loop_time-timesim)*1000.0);
        clock_gettime(CLOCK_REALTIME, &time2);
        timesim = (float)(time2.tv_sec - time1.tv_sec)*1000.0 + (float)(time2.tv_nsec - time1.tv_nsec)/1000000.0;
    }
    if(fabs(timesim - loop_time) > loop_time*0.2)
    {
        printf("Train OverTime lastPeriodTime: %f  loop_time %f \n",timesim,loop_time);//zhy
    }
    }

  //杀死时清理
  datafile.close();
  server.killServer();
}
