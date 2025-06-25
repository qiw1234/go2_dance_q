/*!
 * @file Simulation.h
 * @brief Main simulation class
 */

#ifndef PROJECT_SIMULATION_H
#define PROJECT_SIMULATION_H

#include "ControlParameterInterface.h"
#include "RobotParameters.h"
#include "SimulatorParameters.h"
#include "Dynamics/Uvc_dog.h"
#include "Dynamics/ActuatorModel.h"
#include "Dynamics/Quadruped.h"
#include "Graphics3D.h"
#include "ImuSimulator.h"
#include "SimulatorMessage.h"
#include "SpineBoard.h"
#include "ti_boardcontrol.h"
#include "SharedMemory.h"
#include "utilities/Timer.h"

#include <mutex>
#include <queue>
#include <utility>
#include <vector>
#include <thread>
#include <pthread.h>

#include <lcm/lcm-cpp.hpp>
//#include "simulator_lcmt.hpp"
#if _TC_EN

#undef slots
#include <torch/script.h>
#define slots Q_SLOTS
#endif
#define SIM_LCM_NAME "simulator_state"

/*!
 * Top-level control of a simulation.
 * A simulation includes 1 robot and 1 controller
 * It does not include the graphics window: this must be set with the setWindow
 * method
 */
class Simulation {
  friend class SimControlPanel;
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit Simulation(RobotType robot, Graphics3D* window,
                      SimulatorControlParameters& params, ControlParameters& userParams,
                      std::function<void(void)> ui_update);

  /*!
   * Explicitly set the state of the robot
   */
  void setRobotState(FBModelState<double>& state) {
    _simulator->setState(state);
  }

  void step(double dt, double dtLowLevelControl, double dtHighLevelControl);

  void addCollisionPlane(double mu, double resti, double height,
                         double sizeX = 20, double sizeY = 20,
                         double checkerX = 40, double checkerY = 40,
                         bool addToWindow = true);
  void addCollisionBox(double mu, double resti, double depth, double width,
                       double height, const Vec3<double>& pos,
                       const Mat3<double>& ori, bool addToWindow = true,
                       bool transparent = true);
  void addCollisionMesh(double mu, double resti, double grid_size,
                        const Vec3<double>& left_corner_loc,
                        const DMat<double>& height_map, bool addToWindow = true,
                        bool transparent = true);

  void lowLevelControl();
  void highLevelControl();

  /*!
   * Updates the graphics from the connected window
   */
  void updateGraphics();

  void runAtSpeed(std::function<void(std::string)> error_callback, bool graphics = true);
  void sendControlParameter(const std::string& name,
                            ControlParameterValue value,
                            ControlParameterValueKind kind,
                            bool isUser);

  void resetSimTime() {
    _currentSimTime = 0.;
    _timeOfNextLowLevelControl = 0.;
    _timeOfNextHighLevelControl = 0.;
  }

  ~Simulation() {
    delete _simulator;
    delete _robotDataSimulator;
    delete _imuSimulator;
    //delete _lcm;
  }

  const FBModelState<double>& getRobotState() { return _simulator->getState(); }

  void stop() {
    _running = false;  // kill simulation loop
    _wantStop = true;  // if we're still trying to connect, this will kill us

    if (_connected) {
      _sharedMemory().simToRobot.mode = SimulatorMode::EXIT;
      _sharedMemory().simulatorIsDone();
    }
  }

  SimulatorControlParameters& getSimParams() { return _simParams; }

  RobotControlParameters& getRobotParams() { return _robotParams; }
  ControlParameters& getUserParams() { return _userParams; }

  bool isRobotConnected() { return _connected; }

  void firstRun();
  void buildLcmMessage();
  void loadTerrainFile(const std::string& terrainFileName,
                       bool addGraphics = true);

  bool isGraphicsEnable(){return en_graphics;}
  bool isRunning(){return _running;}
  Graphics3D* getWindow(){return _window;}


 private:
  void handleControlError();
  Graphics3D* _window = nullptr;

  std::mutex _robotMutex;
  SharedMemoryObject<SimulatorSyncronizedMessage> _sharedMemory;
  ImuSimulator<double>* _imuSimulator = nullptr;
  SimulatorControlParameters& _simParams;
  ControlParameters& _userParams;
  RobotControlParameters _robotParams;

  size_t _simRobotID, _controllerRobotID;
  Quadruped<double> _quadruped;
  FBModelState<double> _robotControllerState;
  FloatingBaseModel<double> _model;
  FloatingBaseModel<double> _robotDataModel;
  DVec<double> _tau;
  DynamicsSimulator<double>* _simulator = nullptr;
  DynamicsSimulator<double>* _robotDataSimulator = nullptr;
  std::vector<ActuatorModel<double>> _actuatorModels;
  SpiCommand _spiCommand;
  SpiData _spiData;
#ifdef _USE_ARM
  SpineBoard _spineBoards[_NUM_ARM_SPI+4];
#else
  SpineBoard _spineBoards[4];
#endif
  TI_BoardControl _tiBoards[4];
  RobotType _robot;
  //lcm::LCM* _lcm = nullptr;

  std::function<void(void)> _uiUpdate;
  std::function<void(std::string)> _errorCallback;
  bool _running = false;
  bool _connected = false;
  bool _wantStop = false;
  double _desiredSimSpeed = 1.;
  double _currentSimTime = 0.;
  double _timeOfNextLowLevelControl = 0.;
  double _timeOfNextHighLevelControl = 0.;
  s64 _highLevelIterations = 0;
  //simulator_lcmt _simLCM;
  pthread_t sim_g;
  bool en_graphics;
#ifdef _TC_EN
    typedef struct
    {
        float ocu_xyz[3];    // x y aim angle   z body angle z
        float BodyOrientation[3];   // rot
        float BodyAngularVel[3];
        float jointPos[12];        //
        float jointVel[12];        //
        float jointPosLast0[12];   // tick -3
        float jointPosLast1[12];   // tick -2
        float jointPosLast2[12];   // tick -1
        float jointVelLast0[12];   // tick -2
        float jointVelLast1[12];   // tick -1
        float jointPos_r_Last1[12];// tick -1
        float jointPos_r_Last2[12];// tick
        float baseFreq_;          // ocu_xyz>0 1.3 * freqScale_  double freqScale_ = simulation_dt_ * 2.0 * M_PI;
        float piD_[4];
        float cspi_[8];
        float jointff_r_Last1[12];// tick -1
        float jointff_r_Last2[12];// tick
    }InputData_t;

    typedef struct
    {
        float offset_Freq[4];
        float offset_jointPos_r[12];
        float ff_joint_r[12];
    }OutputData_t;

    Eigen::Vector3d IK_p_tc(size_t leg, const double Z1);
    void Update_Train_Input();
    void processAction();
    Eigen::MatrixXf readMatrix(const char *filename);
#endif
#ifdef _TC_EN
  struct _Joint_action_params
  {
      float kp_;
      float kd_;
      float kff_;
  } Joint_action_params_swing[4][3],Joint_action_params_stand[4][3];
  const float freqScale_;
  char train_first_run;
  Eigen::Matrix<float, 28, 1> actionStd_;
  Eigen::Matrix<float, 28, 1> actionMean_;

  //std::vector<Eigen::Matrix<float, 52, 1>> FHs_;

  torch::jit::script::Module ex_encoder;
  torch::jit::script::Module belief_encoder;
  torch::jit::script::Module actor;
  torch::jit::script::Module belief_decoder;

  std::vector<torch::jit::IValue> ex_encoder_inputs; // 208 -> encoder -> 96

  std::vector<torch::jit::IValue> belief_encoder_inputs; //154 + 96 + hidden -> RNN

  std::vector<torch::jit::IValue> actor_inputs; // RNN->actor-> 28 action

  std::vector<torch::jit::IValue> belief_decoder_inputs; // RNN -> decoder



  Eigen::Matrix<float,154+208,1> E_mean;
  Eigen::Matrix<float,154+208,1> E_var;
  Eigen::Matrix<float, 28, 1> action_t, scaledAction_;
  Eigen::Matrix<double, 12, 1> pTarget_joint_;
  InputData_t input;
  Eigen::Matrix<float,154,1> o_pt_now,o_pt_now_normal;//机体数据 154
  Eigen::Matrix<float,208,1> o_et,o_et_normal; //地形数据 208
  Eigen::Matrix<float, 2, 50> h_t; //rnn隐藏数据
  float pi_[4];
#endif

};

#endif  // PROJECT_SIMULATION_H
