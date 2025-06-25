#if _RAISIM_EN
/*! @file SimulatorMessage.h
 *  @brief Messages sent to/from the development simulator
 *
 *  These messsages contain all data that is exchanged between the robot program
 * and the simulator using shared memory.   This is basically everything except
 * for debugging logs, which are handled by LCM instead
 */

#ifndef PROJECT_SIMULATORTOROBOTMESSAGE_H
#define PROJECT_SIMULATORTOROBOTMESSAGE_H

#include "SharedMemorySim.h"
#include "Eigen/Dense"
//#include "cTypes.h"
//#include "../../common/include/cppTypes.h"
/*!
 * The mode for the simulator
 */
enum class SimulatorMode {
  RUN_CONTROL_PARAMETERS,  // don't run the robot controller, just process
                           // Control Parameters
  RUN_CONTROLLER,          // run the robot controller
  DO_NOTHING,              // just to check connection
  EXIT                     // quit!
};


struct SpiData
{
    float q_abad[4];
    float q_hip[4];
    float q_knee[4];
#if _ARX_ARM_EN
    float q_arm[8];
#endif

    float qd_abad[4];
    float qd_hip[4];
    float qd_knee[4];
#if _ARX_ARM_EN
    float qd_arm[8];
#endif

    float tau_abad[4];
    float tau_hip[4];
    float tau_knee[4];

#if _ARX_ARM_EN
    float tau_arm[8];
#endif
#if _USE_ANKLE || _GOAT
    float q_ankle[4];
    float qd_ankle[4];
    float tau_ankle[4];
#endif
#if _USE_TUO
    float tau_wheel[2];
#endif
#if _USE_ARM_Raisim
    float q_arm[6];
    float qd_arm[6];
    float tau_arm[6];
#endif
    float foot_force[4][3];
};

/*!
 * A plain message from the simulator to the robot
 */

struct SpiCommand {
  float q_des_abad[4];
  float q_des_hip[4];
  float q_des_knee[4];
#if _ARX_ARM_EN
  float q_des_arm[8];
#endif

  float qd_des_abad[4];
  float qd_des_hip[4];
  float qd_des_knee[4];
#if _ARX_ARM_EN
  float qd_des_arm[8];
#endif

  float kp_abad[4];
  float kp_hip[4];
  float kp_knee[4];
#if _ARX_ARM_EN
  float kp_arm[8];
#endif

  float kd_abad[4];
  float kd_hip[4];
  float kd_knee[4];
#if _ARX_ARM_EN
  float kd_arm[8];
#endif

  float tau_abad_ff[4];
  float tau_hip_ff[4];
  float tau_knee_ff[4];
#if _ARX_ARM_EN
  float tau_arm[8];
#endif

#if _USE_ANKLE || _GOAT
    float q_des_ankle[4];
    float qd_des_ankle[4];
    float kp_ankle[4];
    float kd_ankle[4];
    float tau_ankle_ff[4];
#endif

  int32_t flags[4];
#if _USE_GUN
  int32_t shot_flag;
#endif
#if _USE_ARM_Raisim
  float q_des_arm[6];
  float qd_des_arm[6];
  float tau_arm_ff[6];
  float kp_arm[6];
  float kd_arm[6];
#endif
#ifdef _TC_EN
        char action_update_f;
#endif
};


struct SimulatorToRobotMessage {
  Eigen::Matrix<float, 3, 1> omega;
  Eigen::Matrix<float, 4, 1> orientation;
  Eigen::Matrix<float, 3, 1> acc;
  Eigen::Matrix<float, 3, 1> pos;
  Eigen::Matrix<float, 3, 1> vel;
  // leg data
  SpiData spiData;
  SimulatorMode mode;
#if _USE_TUO
  Eigen::Matrix<float, 3, 1> omega_tuo;
  Eigen::Matrix<float, 4, 1> orientation_tuo;
  Eigen::Matrix<float, 3, 1> acc_tuo;
  Eigen::Matrix<float, 3, 1> pos_tuo;
  Eigen::Matrix<float, 3, 1> vel_tuo;
#endif
#if _USE_GUN
  Eigen::Matrix<float, 3, 1> shot_pos_err;
  int shot_flag;
  float shot_R;
  Eigen::Matrix<float, 3, 1> pos_ba;
#endif
};

/*!
 * A plain message from the robot to the simulator
 */
struct RobotToSimulatorMessage {
  SpiCommand spiCommand;
  char errorMessage[2056];
};

/*!
 * All the data shared between the robot and the simulator
 */
struct SimulatorMessage {
  RobotToSimulatorMessage robotToSim;
  SimulatorToRobotMessage simToRobot;
};

/*!
 * A SimulatorSyncronizedMessage is stored in shared memory and is accessed by
 * both the simulator and the robot The simulator and robot take turns have
 * exclusive access to the entire message. The intended sequence is:
 *  - robot: waitForSimulator()
 *  - simulator: *simulates robot* (simulator can read/write, robot cannot do
 * anything)
 *  - simulator: simDone()
 *  - simulator: waitForRobot()
 *  - robot: *runs controller*    (robot can read/write, simulator cannot do
 * anything)
 *  - robot: robotDone();
 *  - robot: waitForSimulator()
 *  ...
 */
struct SimulatorSyncronizedMessage : public SimulatorMessage {

  /*!
   * The init() method should only be called *after* shared memory is connected!
   * This initializes the shared memory semaphores used to keep things in sync
   */
  void init() {
    robotToSimSemaphore.init(0);
    simToRobotSemaphore.init(0);
  }

  /*!
   * Wait for the simulator to respond
   */
  void waitForSimulator() { simToRobotSemaphore.decrement(); }

  /*!
   * Simulator signals that it is done
   */
  void simulatorIsDone() { simToRobotSemaphore.increment(); }

  /*!
   * Wait for the robot to finish
   */
  void waitForRobot() { robotToSimSemaphore.decrement(); }

  /*!
   * Check if the robot is done
   * @return if the robot is done
   */
  bool tryWaitForRobot() { return robotToSimSemaphore.tryDecrement(); }

  /*!
   * Wait for the robot to finish with a timeout
   * @return if we finished before timing out
   */
  bool waitForRobotWithTimeout() {
    return robotToSimSemaphore.decrementTimeout(1, 0);
  }

  /*!
   * Signal that the robot is done
   */
  void robotIsDone() { robotToSimSemaphore.increment(); }

 private:
  SharedMemorySemaphore robotToSimSemaphore, simToRobotSemaphore;
};

#endif  // PROJECT_SIMULATORTOROBOTMESSAGE_H
#endif
