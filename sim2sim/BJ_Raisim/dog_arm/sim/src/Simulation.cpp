#include "Simulation.h"
#include "Dynamics/Quadruped.h"
#include "ParamYaml.h"

#include <Configuration.h>
//#include <include/GameController.h>
#include <unistd.h>
#include <fstream>
#include <sys/timerfd.h>

using namespace std;

// if DISABLE_HIGH_LEVEL_CONTROL is defined, the simulator will run freely,
// without trying to connect to a robot
//#define DISABLE_HIGH_LEVEL_CONTROL


/*!
 * Initialize the simulator here.  It is _not_ okay to block here waiting for
 * the robot to connect. Use firstRun() instead!
 */

Simulation::Simulation(RobotType robot, Graphics3D* window,
                       SimulatorControlParameters& params, ControlParameters& userParams, std::function<void(void)> uiUpdate)
    : _simParams(params), _userParams(userParams), _tau(12)
    #ifdef _TC_EN
    ,freqScale_( 0.002 * 2.0 * M_PI)
    #endif
{
  _uiUpdate = uiUpdate;
  // init parameters
  printf("[Simulation] Load parameters...\n");
  _simParams
      .lockMutex();  // we want exclusive access to the simparams at this point
  if (!_simParams.isFullyInitialized()) {
    printf(
        "[ERROR] Simulator parameters are not fully initialized.  You forgot: "
        "\n%s\n",
        _simParams.generateUnitializedList().c_str());
    throw std::runtime_error("simulator not initialized");
  }

#if 0//no use
  // init LCM
  if (_simParams.sim_state_lcm) {
    printf("[Simulation] Setup LCM...\n");
    _lcm = new lcm::LCM(getLcmUrl(_simParams.sim_lcm_ttl));
    if (!_lcm->good()) {
      printf("[ERROR] Failed to set up LCM\n");
      throw std::runtime_error("lcm bad");
    }
  }
#endif
  // init quadruped info
  printf("[Simulation] Build quadruped...\n");
  _robot = robot;
  _quadruped = buildUvcDog<double>();
  printf("[Simulation] Build actuator model...\n");

  _actuatorModels = _quadruped.buildActuatorModels();
  _window = window;

  // init graphics
  if (_window) {
    printf("[Simulation] Setup Cheetah graphics...\n");
    Vec4<float> truthColor, seColor;
//    truthColor << 0.2, 0.4, 0.2, 0.6;
//    seColor << .75,.75,.75, 1.0;

    truthColor << 0.2, 0.4, 0.2, 0.6;//dyn --xp
    seColor << 0.0, 1.0, 0.0, 1.0;//graphic --xp
    _simRobotID = window->setupMiniCheetah(truthColor, true, false);//true
    //_controllerRobotID = window->setupMiniCheetah(seColor, false, false);

    //window->setupSphereTest(0.2);



#ifdef _TC_EN0
    actionStd_<<(float)0.5 * freqScale_, (float)0.5 * freqScale_, (float)0.5 * freqScale_, (float)0.5 * freqScale_,
            M_PI/10, M_PI/6, M_PI/6,
            M_PI/10, M_PI/6, M_PI/6,
            M_PI/10, M_PI/6, M_PI/6,
            M_PI/10, M_PI/6, M_PI/6,
            1.0,   1.0,   1.0,
            1.0,   1.0,   1.0,
            1.0,   1.0,   1.0,
            1.0,   1.0,   1.0;
    actionMean_<< 0.0, 0.0, 0.0, 0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,
            0.0,     0.0,     0.0;
    //FHs_.resize(4);
    ex_encoder_inputs.resize(1);
    belief_encoder_inputs.resize(3);
    actor_inputs.resize(3);
    belief_decoder_inputs.resize(2);
    o_pt_now.setZero(154);
    o_et.setZero(208);
    h_t.setZero();
    E_mean = readMatrix("mean.csv");
    E_var = readMatrix("var.csv");
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        ex_encoder = torch::jit::load("ex_encoder.pt");
        belief_encoder = torch::jit::load("belief_encoder.pt");
        actor = torch::jit::load("actor.pt");
        belief_decoder = torch::jit::load("belief_decoder.pt");
      }
      catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
      }


    for(int i = 0;i<4;i++)
    {
//                    Joint_action_params_swing[i][0].kp_ = 100.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][0].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][0].kff_ = 50.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][1].kp_ = 100.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][1].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][1].kff_ = 70.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][2].kp_ = 50.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][2].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][2].kff_ = 150.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_swing[i][0].kp_ = 30.0;//300.0;//150.0;//150.0f;
        Joint_action_params_swing[i][0].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_swing[i][0].kff_ = 50.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_swing[i][1].kp_ = 30.0;//300.0;//150.0;//150.0f;
        Joint_action_params_swing[i][1].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_swing[i][1].kff_ = 70.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_swing[i][2].kp_ = 30.0;//300.0;//150.0;//150.0f;
        Joint_action_params_swing[i][2].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_swing[i][2].kff_ = 150.0;//2.5;//1.3f;//1.3;//3.0;//1.3;

        Joint_action_params_stand[i][0].kp_ = 30.0;//300.0;//150.0;//150.0f;
        Joint_action_params_stand[i][0].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_stand[i][0].kff_ = 50.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_stand[i][1].kp_ = 30.0;//300.0;//150.0;//150.0f;
        Joint_action_params_stand[i][1].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_stand[i][1].kff_ = 70.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_stand[i][2].kp_ = 30.0;//300.0;//150.0;//150.0f;
        Joint_action_params_stand[i][2].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
        Joint_action_params_stand[i][2].kff_ = 150.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
    }
    train_first_run = 1;


#endif



  }

  // init rigid body dynamics
  printf("[Simulation] Build rigid body model...\n");
  _model = _quadruped.buildModel();
  _robotDataModel = _quadruped.buildModel();
  _simulator =
      new DynamicsSimulator<double>(_model, (bool)_simParams.use_spring_damper);
  double mass_all  = _model.totalNonRotorMass();
  printf("#######mass_all########%f\n",mass_all);
  //_robotDataSimulator = new DynamicsSimulator<double>(_robotDataModel, false);
//#ifdef _WHEEL_EN
//  DVec<double> zero12(16);
//  for (u32 i = 0; i < 16; i++) {
//    zero12[i] = 0.;
//  }
//#else
//  DVec<double> zero12(12);
//  for (u32 i = 0; i < 12; i++) {
//    zero12[i] = 0.;
//  }
//#endif
  DVec<double> zero12(12);
#ifdef _WHEEL_EN
    zero12.resize(16);
#elif _USE_ARM
    zero12.resize(12+_DOF_ARM);
#else

#endif
  zero12.setZero();
  // set some sane defaults:
  _tau = zero12;
  _robotControllerState.q = zero12;
  _robotControllerState.qd = zero12;
  FBModelState<double> x0;
  x0.bodyOrientation = rotationMatrixToQuaternion(
      ori::coordinateRotation(CoordinateAxis::Z, 0.01));
  // Mini Cheetah
  x0.bodyPosition.setZero();
  x0.bodyVelocity.setZero();
  x0.q = zero12;
  x0.qd = zero12;

  // Mini Cheetah Initial Posture
  // x0.bodyPosition[2] = -0.49;
  // Cheetah 3
  x0.bodyPosition[0] = -0.15;
  x0.bodyPosition[2] = 0.59;
  //jump haogou
 // x0.bodyPosition[0] = -0.1;
//   x0.bodyPosition[2] = 0.59;

#ifdef _WHEEL_EN
   x0.q[0] = -0.5;//0.0;//-0.807;
   x0.q[1] = -2.0;//-2.4;
   x0.q[2] = 1.57;
   x0.q[3] = 0;

   x0.q[4] = 0.5;//0;//0.807;
   x0.q[5] = -2.0;//-2.4;
   x0.q[6] = 1.57;
   x0.q[7] = 0;

   x0.q[8] = -0.5;//0;//-0.807;
   x0.q[9] = -2.0;//-2.4;
   x0.q[10] = 1.57;
   x0.q[11] = 0;

   x0.q[12] = 0.5;//0;//0.807;
   x0.q[13] = -2.0;//-2.4;
   x0.q[14] = 1.57;
   x0.q[15] = 0;
#elif _USE_ARM
   x0.q[0] = -0.5;//0.0;//-0.807;
   x0.q[1] = -2.0;//-2.4;
   x0.q[2] = 1.57;

   x0.q[3] = 0.5;//0;//0.807;
   x0.q[4] = -2.0;//-2.4;
   x0.q[5] = 1.57;

   x0.q[6] = -0.5;//0;//-0.807;
   x0.q[7] = -2.0;//-2.4;
   x0.q[8] = 1.57;

   x0.q[9] = 0.5;//0;//0.807;
   x0.q[10] = -2.0;//-2.4;
   x0.q[11] = 1.57;
   for(int i =12;i<_DOF_ARM;i++)
       x0.q[i] = 0;
#else
//   x0.q[0] = -0.5;//0.0;//-0.807;
//   x0.q[1] = -2.0;//-2.4;
//   x0.q[2] = 1.57;

//   x0.q[3] = 0.5;//0;//0.807;
//   x0.q[4] = -2.0;//-2.4;
//   x0.q[5] = 1.57;

//   x0.q[6] = -0.5;//0;//-0.807;
//   x0.q[7] = -2.0;//-2.4;
//   x0.q[8] = 1.57;

//   x0.q[9] = 0.5;//0;//0.807;
//   x0.q[10] = -2.0;//-2.4;
//   x0.q[11] = 1.57;

   //jump haogou
   x0.q[0] = -0.0854;//0.0;//-0.807;
   x0.q[1] = -1.0835;//-2.4;
   x0.q[2] = 2.4778;

   x0.q[3] = 0.0854;//0;//0.807;
   x0.q[4] = -1.0835;//-2.4;
   x0.q[5] = 2.4778;

   x0.q[6] = -0.0854;//0;//-0.807;
   x0.q[7] = -1.0835;//-2.4;
   x0.q[8] = 2.4778;

   x0.q[9] = 0.0854;//0;//0.807;
   x0.q[10] = -1.0835;//-2.4;
   x0.q[11] = 2.4778;

#endif

  // Initial (Mini Cheetah stand)
  // x0.bodyPosition[2] = -0.185;
  // Cheetah 3
  // x0.bodyPosition[2] = -0.075;

  // x0.q[0] = -0.03;
  // x0.q[1] = -0.79;
  // x0.q[2] = 1.715;

  // x0.q[3] = 0.03;
  // x0.q[4] = -0.79;
  // x0.q[5] = 1.715;

  // x0.q[6] = -0.03;
  // x0.q[7] = -0.72;
  // x0.q[8] = 1.715;

  // x0.q[9] = 0.03;
  // x0.q[10] = -0.72;
  // x0.q[11] = 1.715;

  // Cheetah lies on the ground
  //x0.bodyPosition[2] = -0.45;
//  x0.bodyPosition[2] = 0.05;//body height
//  x0.q[0] = -0.7;//rf leg joint 1
//  x0.q[1] = 1.;//
//  x0.q[2] = 2.715;

//  x0.q[3] = 0.7;
//  x0.q[4] = 1.;
//  x0.q[5] = 2.715;

//  x0.q[6] = -0.7;
//  x0.q[7] = -1.0;
//  x0.q[8] = -2.715;

//  x0.q[9] = 0.7;
//  x0.q[10] = -1.0;
//  x0.q[11] = -2.715;


  setRobotState(x0);
  //_robotDataSimulator->setState(x0);

  printf("[Simulation] Setup low-level control...\n");
  // init spine:
  if (_robot == RobotType::Uvc_dog) {
    for (int leg = 0; leg < 4; leg++) {
      _spineBoards[leg].init(Quadruped<float>::getSideSign(leg), leg);
      _spineBoards[leg].data = &_spiData;
      _spineBoards[leg].cmd = &_spiCommand;
      _spineBoards[leg].resetData();
      _spineBoards[leg].resetCommand();
    }
#ifdef _USE_ARM
    for (int arm = 4; arm < 4+_NUM_ARM_SPI; arm++) {
      _spineBoards[arm].init(1, arm);
      _spineBoards[arm].data = &_spiData;
      _spineBoards[arm].cmd = &_spiCommand;
      _spineBoards[arm].resetData();
      _spineBoards[arm].resetCommand();
    }
#endif
  }

  // init shared memory
  printf("[Simulation] Setup shared memory...\n");
  _sharedMemory.createNew(DEVELOPMENT_SIMULATOR_SHARED_MEMORY_NAME, true);
  _sharedMemory().init();

  // shared memory fields:
  _sharedMemory().simToRobot.robotType = _robot;
  _window->_drawList._visualizationData =
      &_sharedMemory().robotToSim.visualizationData;

  // load robot control parameters
  printf("[Simulation] Load control parameters...\n");
  if (_robot == RobotType::Uvc_dog) {
    _robotParams.initializeFromYamlFile(getConfigDirectoryPath() +
                                        MINI_CHEETAH_DEFAULT_PARAMETERS);
  }

  if (!_robotParams.isFullyInitialized()) {
    printf("Not all robot control parameters were initialized. Missing:\n%s\n",
           _robotParams.generateUnitializedList().c_str());
    throw std::runtime_error("not all parameters initialized from ini file");
  }
  // init IMU simulator
  printf("[Simulation] Setup IMU simulator...\n");
  _imuSimulator = new ImuSimulator<double>(_simParams);

  _simParams.unlockMutex();
  printf("[Simulation] Ready!\n");
}

void Simulation::sendControlParameter(const std::string& name,
                                      ControlParameterValue value,
                                      ControlParameterValueKind kind, bool isUser) {
  ControlParameterRequest& request =
      _sharedMemory().simToRobot.controlParameterRequest;
  ControlParameterResponse& response =
      _sharedMemory().robotToSim.controlParameterResponse;

  // first check no pending message
  assert(request.requestNumber == response.requestNumber);

  // new message
  request.requestNumber++;

  // message data
  request.requestKind = isUser ? ControlParameterRequestKind::SET_USER_PARAM_BY_NAME : ControlParameterRequestKind::SET_ROBOT_PARAM_BY_NAME;
  strcpy(request.name, name.c_str());
  request.value = value;
  //cout << "value = " << value.d << endl;
  //printf("value = %ld\n", value.i);
  request.parameterKind = kind;
  printf("%s\n", request.toString().c_str());

  // run robot:
  _robotMutex.lock();
  _sharedMemory().simToRobot.mode = SimulatorMode::RUN_CONTROL_PARAMETERS;
  _sharedMemory().simulatorIsDone();

  // wait for robot code to finish
  if (_sharedMemory().waitForRobotWithTimeout()) {
  } else {
    handleControlError();
    request.requestNumber = response.requestNumber; // so if we come back we won't be off by 1
    _robotMutex.unlock();
    return;
  }

  //_sharedMemory().waitForRobot();
  _robotMutex.unlock();

  // verify response is good
  assert(response.requestNumber == request.requestNumber);
  assert(response.parameterKind == request.parameterKind);
  assert(std::string(response.name) == request.name);
}

/*!
 * Report a control error.  This doesn't throw and exception and will return so you can clean up
 */
void Simulation::handleControlError() {
  _wantStop = true;
  _running = false;
  _connected = false;
  _uiUpdate();
  if(!_sharedMemory().robotToSim.errorMessage[0]) {
    printf(
      "[ERROR] Control code timed-out!\n");
    _errorCallback("Control code has stopped responding without giving an error message.\nIt has likely crashed - "
                   "check the output of the control code for more information");

  } else {
    printf("[ERROR] Control code has an error!\n");
    _errorCallback("Control code has an error:\n" + std::string(_sharedMemory().robotToSim.errorMessage));
  }

}

/*!
 * Called before the simulator is run the first time.  It's okay to put stuff in
 * here that blocks on having the robot connected.
 */
void Simulation::firstRun() {
  // connect to robot
  _robotMutex.lock();
  _sharedMemory().simToRobot.mode = SimulatorMode::DO_NOTHING;
  _sharedMemory().simulatorIsDone();

  printf("[Simulation] Waiting for robot...\n");

  // this loop will check to see if the robot is connected at 10 Hz
  // doing this in a loop allows us to click the "stop" button in the GUI
  // and escape from here before the robot code connects, if needed
  while (!_sharedMemory().tryWaitForRobot()) {
    if (_wantStop) {
      return;
    }
    usleep(100000);
  }
  printf("Success! the robot is alive\n");
  _connected = true;
  _uiUpdate();
  _robotMutex.unlock();

  // send all control parameters
  printf("[Simulation] Send robot control parameters to robot...\n");
//  for (auto& kv : _robotParams.collection._map) {
//    sendControlParameter(kv.first, kv.second->get(kv.second->_kind),
//                         kv.second->_kind, false);
//  }

//  for (auto& kv : _userParams.collection._map) {
//    sendControlParameter(kv.first, kv.second->get(kv.second->_kind),
//                         kv.second->_kind, true);
//  }

  printf("firstRun end\n");
}

/*!
 * Take a single timestep of dt seconds
 */
void Simulation::step(double dt, double dtLowLevelControl,
                      double dtHighLevelControl) {
  // Low level control (if needed)
  if (_currentSimTime >= _timeOfNextLowLevelControl) {
    lowLevelControl();
    _timeOfNextLowLevelControl = _timeOfNextLowLevelControl + dtLowLevelControl;
  }

  // High level control
  if (_currentSimTime >= _timeOfNextHighLevelControl) {
#ifndef DISABLE_HIGH_LEVEL_CONTROL
    highLevelControl();
#endif
    _timeOfNextHighLevelControl =
        _timeOfNextHighLevelControl + dtHighLevelControl;
  }

  // actuator model:
  if (_robot == RobotType::Uvc_dog) {
    for (int leg = 0; leg < 4; leg++) {
#ifdef _WHEEL_EN
        for (int joint = 0; joint < 4; joint++) {
          _tau[leg * 4 + joint] = _actuatorModels[joint].getTorque(
              _spineBoards[leg].torque_out[joint],
              _simulator->getState().qd[leg * 4 + joint]);
        }
#else
        for (int joint = 0; joint < 3; joint++) {
          _tau[leg * 3 + joint] = _actuatorModels[joint].getTorque(
              _spineBoards[leg].torque_out[joint],
              _simulator->getState().qd[leg * 3 + joint]);


          if(leg==0&&joint==2)
          {
              //printf("[%d][%d]tau = %f,%f\n", leg,joint,_tau[leg * 3 + joint],_spineBoards[leg].torque_out[joint]);
          }

            //_tau[leg * 3 + joint] = 0.0f;
        }
#ifdef _TC_EN0
        if(_spiCommand.action_update_f == 1)
        {
            int k = 0;
            switch (leg) {
            case 0:
                k=1;
                break;
            case 1:
                k=0;
                break;
            case 2:
                k=3;
                break;
            case 3:
                k=2;
                break;
            }
            for (int joint = 0; joint < 3; joint++) {
                char f = 1;
                if(joint>0)
                    f = -1;
                else
                    f = 1;
              _tau[leg * 3 + joint] = Joint_action_params_swing[k]->kp_ *
                      (f*pTarget_joint_[k*3+joint] - _simulator->getState().q[leg * 3 + joint]) +
                  Joint_action_params_swing[k]->kd_ * (0 -
                                             _simulator->getState().qd[leg * 3 + joint]) +
                  Joint_action_params_swing[k]->kff_ * f * scaledAction_[16+k*3+joint];
            }
        }
#endif
#endif

    }
#ifdef _USE_ARM
    for (int arm = 4; arm < 4+_NUM_ARM_SPI; arm++){
        for(int arm_link = 0;arm_link<3;arm_link++){
            if(((arm-4)*3 + arm_link)>_DOF_ARM-1)
                break;
        _tau[12 +(arm-4)*3+ arm_link] = _actuatorModels[3].getTorque(
        _spineBoards[arm].torque_out[arm_link],
        _simulator->getState().qd[12 +(arm-4)*3+ arm_link]);
        }
    }
#endif

  }

  // dynamics
  _currentSimTime += dt;

  // Set Homing Information
  RobotHomingInfo<double> homing;
  homing.active_flag = _simParams.go_home;
  homing.position = _simParams.home_pos;
  homing.rpy = _simParams.home_rpy;
  homing.kp_lin = _simParams.home_kp_lin;
  homing.kd_lin = _simParams.home_kd_lin;
  homing.kp_ang = _simParams.home_kp_ang;
  homing.kd_ang = _simParams.home_kd_ang;
  _simulator->setHoming(homing);

  //cout << "_tau " << _tau << endl;
  _simulator->step(dt, _tau, _simParams.floor_kp, _simParams.floor_kd);
}

void Simulation::lowLevelControl() {
  if (_robot == RobotType::Uvc_dog) {
    // update spine board data:
    for (int leg = 0; leg < 4; leg++) {



#ifdef _WHEEL_EN
        _spiData.q_abad[leg] = _simulator->getState().q[leg * 4 + 0];
        _spiData.q_hip[leg] = _simulator->getState().q[leg * 4 + 1];
        _spiData.q_knee[leg] = _simulator->getState().q[leg * 4 + 2];
        _spiData.q_wheel[leg] = _simulator->getState().q[leg * 4 + 3];

        //printf("%f %f %f\t", _spiData.q_abad[leg], _spiData.q_hip[leg], _spiData.q_knee[leg]);//---scs

        _spiData.qd_abad[leg] = _simulator->getState().qd[leg * 4 + 0];
        _spiData.qd_hip[leg] = _simulator->getState().qd[leg * 4 + 1];
        _spiData.qd_knee[leg] = _simulator->getState().qd[leg * 4 + 2];
        _spiData.q_wheel[leg] = _simulator->getState().qd[leg * 4 + 3];


        _spiData.tau_abad[leg] = _tau[leg * 4 + 0];//_spineBoards[leg].torque_out[0];//
        _spiData.tau_hip[leg] = _tau[leg * 4 + 1];//_spineBoards[leg].torque_out[1];//
        _spiData.tau_knee[leg] = _tau[leg * 4 + 2];//_spineBoards[leg].torque_out[2];//
        _spiData.tau_knee[leg] = _tau[leg * 4 + 3];
#else
        _spiData.q_abad[leg] = _simulator->getState().q[leg * 3 + 0];
        _spiData.q_hip[leg] = _simulator->getState().q[leg * 3 + 1];
        _spiData.q_knee[leg] = _simulator->getState().q[leg * 3 + 2];

        //printf("%f %f %f\t", _spiData.q_abad[leg], _spiData.q_hip[leg], _spiData.q_knee[leg]);//---scs

        _spiData.qd_abad[leg] = _simulator->getState().qd[leg * 3 + 0];
        _spiData.qd_hip[leg] = _simulator->getState().qd[leg * 3 + 1];
        _spiData.qd_knee[leg] = _simulator->getState().qd[leg * 3 + 2];


        _spiData.tau_abad[leg] = _tau[leg * 3 + 0];//_spineBoards[leg].torque_out[0];//
        _spiData.tau_hip[leg] = _tau[leg * 3 + 1];//_spineBoards[leg].torque_out[1];//
        _spiData.tau_knee[leg] = _tau[leg * 3 + 2];//_spineBoards[leg].torque_out[2];//
#endif

      size_t gcID = _simulator->getModel()._footIndicesGC.at(leg);
      for(int vec = 0;vec<3;vec++)
        _spiData.foot_force[leg][vec] = _simulator->getModel()._pGC.at(gcID)[vec];
      //printf("%f %f %f\t", _spiData.qd_abad[leg], _spiData.qd_hip[leg], _spiData.qd_knee[leg]);//---scs
    }
    //printf("\n");
#ifdef _USE_ARM
    for (int arm_link = 0; arm_link < _DOF_ARM; arm_link++)
    {
        _spiData.q_arm[arm_link] = _simulator->getState().q[12+arm_link];
        _spiData.qd_arm[arm_link] = _simulator->getState().qd[12+arm_link];
        _spiData.tau_arm[arm_link] = _tau[12+arm_link];
    }
#endif
    // run spine board control:
    for (auto& spineBoard : _spineBoards) {
      spineBoard.run();
    }

  }
}

#ifdef _TC_EN0

Eigen::MatrixXf Simulation::readMatrix(const char *filename)
    {
    int cols = 0, rows = 0;
    double buff[(int) 1e6];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
        {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
        }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXf result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;}





Eigen::Vector3d Simulation::IK_p_tc(size_t leg, const double Z1){

   const double a1 = 0;
   const double a2 = 0.34;
   const double d = 0.10575;
   const double a3 = sqrt(0.37151 * 0.37151 + 0.01478 * 0.01478);
   double h0_ = 0.55;

   Eigen::Vector3d joint_p_d,f_d_W,f_d_B;

   float k_leg_flag = 1.0f;
   float k_leg_flag_w = 1.0f;

   if(leg==2||leg==3) k_leg_flag*=-1.0f;//
   if(leg==1||leg==3) k_leg_flag_w*=-1.0f;//


   // foot_pos_hip(0) = -(0.5);// + S1 * d * k_leg_flag_w;
   // foot_pos_hip(1) = 0;//foot_p_d(1) - k_leg_flag_w*body_w/2.0f;// - C1 * d * k_leg_flag_w;
   // foot_pos_hip(2) = 0;//foot_p_d(0) - k_leg_flag*body_l/2.0f;
   f_d_W << 0,0, h0_ - Z1;
   f_d_B = quaternionToRotationMatrix(_sharedMemory().simToRobot.cheaterState.orientation).transpose()*f_d_W;
   double px = f_d_B[2];
   double py = -f_d_B[1];
   double pz = -f_d_B[0];

   //double solved_q_d[3];
   double pi = M_PI;
   double pyz_new = sqrt(py*py + pz*pz);
   double pxz_new = sqrt(px*px + pz*pz);
   double pxyz_new = sqrt(px*px + py*py + pz*pz);
   double a_sin_d = d/std::sqrt(px*px+py*py);
   if(a_sin_d>=1.0f) a_sin_d = 1.0f;
   if(a_sin_d<=-1.0f) a_sin_d = -1.0f;

   joint_p_d[0] = std::atan2(py,px) - std::asin(a_sin_d) * k_leg_flag_w;//p desred


   px = px + std::sin(joint_p_d[0])* d * k_leg_flag_w;
   py = py - std::cos(joint_p_d[0]) * d * k_leg_flag_w;

   double tmp_v1 = std::sqrt(px*px+py*py)-a1;
   double tmp_v = (tmp_v1)*(tmp_v1)+pz*pz;

   if(k_leg_flag_w > 0) joint_p_d[0] = std::max(std::min(joint_p_d[0],0.80),-0.64);
   if(k_leg_flag_w < 0) joint_p_d[0] = std::max(std::min(joint_p_d[0],0.64),-0.80);

   double a_c = (tmp_v - a2*a2 - a3*a3)/(2.0f*a2*a3);//(a2*a2+a3*a3-tmp_v)/(2.0f*a2*a3);
   if(a_c>=1.0f) a_c = 1.0f;
   if(a_c<=-1.0f) a_c = -1.0f;
   joint_p_d(2) = std::acos(a_c);//k_leg_flag*(pi-acos(a_c));

   double theta_p1 = atan2(pz,-tmp_v1);
   double a_c1 = (a2*a2-a3*a3+tmp_v)/(2.0f*a2*sqrt(tmp_v));
   if(a_c1>=1.0f) a_c1 = 1.0f;
   if(a_c1<=-1.0f) a_c1 = -1.0f;
   double q1_tmp = acos(a_c1);
   double q01 = q1_tmp + theta_p1;
   double q02 = -q1_tmp + theta_p1;

   if(pz<0.0f) joint_p_d(1) = q01 + pi;
   if(pz>=0.0f) joint_p_d(1) = q01 - pi;
   if(joint_p_d(1)>=2.0*pi) joint_p_d(1)-=2.0*pi;
   if(joint_p_d(1)<=-2.0*pi) joint_p_d(1)+=2.0*pi;
   joint_p_d(2)*=-1.0f;
   joint_p_d[2]+= (-1.0) * std::atan2(0.01478,0.37151);

   joint_p_d[1] = std::max(std::min(joint_p_d[1],2.61),-0.52);
   joint_p_d[2] = std::max(std::min(joint_p_d[2],-0.52),-2.61);
   // std::cout << joint_p_d << std::endl;
   return joint_p_d;
 }

inline int fastfloor(double a) {
  int i = int(a);
  if (i > a) i--;
  return i;
}
inline double wrapAngle(double a) {
  double twopi = 2.0 * M_PI;
  return a - twopi * fastfloor(a / twopi);
}
inline double anglemod(double a) {
    return wrapAngle((a + M_PI)) - M_PI;
}





void Simulation::processAction() {
   double clearance_[4];
   int nJoints_ = 12;
    scaledAction_ = action_t.cwiseProduct(actionStd_) + actionMean_;

    for (size_t i = 0; i < 4; i++) {
      input.piD_[i] = scaledAction_[i] + input.baseFreq_;
    }

    for (size_t i = 0; i < 4; i++) {
          clearance_[i] = 0.18;
        }

    /// update Target
    float simulation_dt_ = 0.001, control_dt_ = 0.02;
    size_t decimation = (size_t) (control_dt_ / simulation_dt_);

    for (size_t j = 0; j < 4; j++) {
      pi_[j] += input.piD_[j] * decimation;
      pi_[j] = anglemod(pi_[j]);
    }
    Eigen::Vector3d desired_angle;
    for (size_t j = 0; j < 4; j++) {
      double dh = 0.0;
      if (pi_[j] > 0.0) {
        double t = pi_[j] / M_PI_2;
        if (t < 1.0) {
          double t2 = t * t;
          double t3 = t2 * t;
          dh = (-2 * t3 + 3 * t2);
        } else {
          t = t - 1;
          double t2 = t * t;
          double t3 = t2 * t;
          dh = (2 * t3 - 3 * t2 + 1.0);
        }
        dh *= clearance_[j];
      }
      desired_angle = IK_p_tc(j, dh);
      pTarget_joint_.segment<3>(3 * j) = desired_angle;

      pTarget_joint_(3 * j) += scaledAction_.tail(nJoints_*2)[3 * j];
      pTarget_joint_(3 * j + 1) += scaledAction_.tail(nJoints_*2)[3 * j + 1];
      pTarget_joint_(3 * j + 2) += scaledAction_.tail(nJoints_*2)[3 * j + 2];;
    }
    static int tick =0;
//    if(tick<3)
//        std::cout<<"############target##########"<<pTarget_joint_<<std::endl;
    tick++;
}
void Simulation::Update_Train_Input()
{
    input.ocu_xyz[0] = 0;
    input.ocu_xyz[1] = 0;
    input.ocu_xyz[2] = 0;
    for(int i = 0; i < 3; i++)
    {
        input.BodyOrientation[i] = quaternionToRotationMatrix(_sharedMemory().simToRobot.cheaterState.orientation).row(2)[i];
        input.BodyAngularVel[i] = _sharedMemory().simToRobot.cheaterState.omegaBody[i];
    }
    for(int i = 0; i < 4; i++)
    {
        int k = 0;
        switch (i) {
        case 0:
            k=1;
            break;
        case 1:
            k=0;
            break;
        case 2:
            k=3;
            break;
        case 3:
            k=2;
            break;
        }
        for(int j = 0; j < 3; j++)
        {
            input.jointPosLast0[i*3+j] = input.jointPosLast1[i*3+j];
            input.jointPosLast1[i*3+j] = input.jointPosLast2[i*3+j];
            input.jointPosLast2[i*3+j] = input.jointPos[i*3+j];

            input.jointVelLast0[i*3+j] = input.jointVelLast1[i*3+j];
            input.jointVelLast1[i*3+j] = input.jointVel[i*3+j];

            if(j == 0)
            {
                input.jointPos[i*3+j] = _spiData.q_abad[k];
                input.jointVel[i*3+j] = _spiData.qd_abad[k];
            }
            else if(j == 1)
            {
                input.jointPos[i*3+j] = -_spiData.q_hip[k];
                input.jointVel[i*3+j] = -_spiData.qd_hip[k];
            }
            else if(j == 2)
            {
                input.jointPos[i*3+j] = -_spiData.q_knee[k];
                input.jointVel[i*3+j] = -_spiData.qd_knee[k];
            }

            input.jointPos_r_Last1[i*3+j] = input.jointPos_r_Last2[i*3+j];
            input.jointPos_r_Last2[i*3+j] = pTarget_joint_[i*3+j];

            input.jointff_r_Last1[i*3+j] = input.jointff_r_Last2[i*3+j];
            input.jointff_r_Last2[i*3+j] = scaledAction_[16+i*3+j];
        }
        input.cspi_[i*2] = sin(pi_[i]);
        input.cspi_[i*2 + 1] = cos(pi_[i]);

        float height = ((double)random()/(double)RAND_MAX)*0.0;//foot_pos_ground[i][2]+_sharedMemory().simToRobot.cheaterState.position[2] - 0.041;
        if(height > 1)
            height = 1;
        else if(height < -1)
            height = -1;
        o_et.segment<52>(i * 52) = Eigen::Matrix<float, 52, 1>::Ones()*height;
    }
    if(fabs(input.ocu_xyz[0])>0.000001||fabs(input.ocu_xyz[1])>0.000001||fabs(input.ocu_xyz[2])>0.000001)
    {
        input.baseFreq_ = 1.3 * freqScale_;
    }
    else
    {
        input.baseFreq_ = 0 ;
    }
    float * pinput = (float *)&input;
    for(int i=0;i<154;i++)
    {
        o_pt_now[i] = pinput[i];
    }

}
#endif
void Simulation::highLevelControl() {
  // send joystick data to robot:
//  _sharedMemory().simToRobot.gamepadCommand = _window->getDriverCommand();
//  _sharedMemory().simToRobot.gamepadCommand.applyDeadband(
//      _simParams.game_controller_deadband);

  // send IMU data to robot:
  _imuSimulator->updateCheaterState(_simulator->getState(),
                                    _simulator->getDState(),
                                    _sharedMemory().simToRobot.cheaterState);

  _imuSimulator->updateVectornav(_simulator->getState(),
                                   _simulator->getDState(),
                                   &_sharedMemory().simToRobot.vectorNav);


  // send leg data to robot
  if (_robot == RobotType::Uvc_dog) {
    _sharedMemory().simToRobot.spiData = _spiData;
  }else {
    assert(false);
  }

  // signal to the robot that it can start running
  // the _robotMutex is used to prevent qt (which runs in its own thread) from
  // sending a control parameter while the robot code is already running.
  _robotMutex.lock();
  _sharedMemory().simToRobot.mode = SimulatorMode::RUN_CONTROLLER;
  _sharedMemory().simulatorIsDone();

  // wait for robot code to finish (and send LCM while waiting)
#if 0
  if (_lcm) {
    buildLcmMessage();
    _lcm->publish(SIM_LCM_NAME, &_simLCM);
  }
#endif //no use

  // first make sure we haven't killed the robot code
  if (_wantStop) return;

  // next try waiting at most 1 second:                //qiu
//  if (_sharedMemory().waitForRobotWithTimeout()) {
//  } else {
//    handleControlError();
//    _robotMutex.unlock();
//    return;
//  }
  _robotMutex.unlock();

  // update  from robot ---xp
  if (_robot == RobotType::Uvc_dog) {
    _spiCommand = _sharedMemory().robotToSim.spiCommand;
  }
#ifdef _TC_EN
//  static struct timespec time1 = {0, 0};
//  static struct timespec time2 = {0, 0};
//  static float timesim =0 ;
//  static char flag_train = -1;
//  if(_spiCommand.action_update_f == 1 && flag_train == -1)
//  {
//      clock_gettime(CLOCK_REALTIME, &time2);
//      timesim = (float)(time2.tv_sec - time1.tv_sec)*1000.0 + (float)(time2.tv_nsec - time1.tv_nsec)/1000000.0 ;
//      printf("###################time: %f ms#####################\n",timesim);
//      clock_gettime(CLOCK_REALTIME, &time1);
//      flag_train = 1;
//  }
//  else if(_spiCommand.action_update_f == -1 && flag_train == 1)
//  {
//      flag_train = -1;
//  }
//  printf("#####update: %d#######\n",_spiCommand.action_update_f);

  //train_update
#ifdef _TC_EN0
  static int tick = 0;
  static struct timespec time1 = {0, 0};
  static struct timespec time2 = {0, 0};
  static float timesim =0 ;
if(_spiCommand.action_update_f == 1)
{
    if(train_first_run == 1)
    {
        actionStd_<<(float)0.5 * freqScale_, (float)0.5 * freqScale_, (float)0.5 * freqScale_, (float)0.5 * freqScale_,
                M_PI/10, M_PI/6, M_PI/6,
                M_PI/10, M_PI/6, M_PI/6,
                M_PI/10, M_PI/6, M_PI/6,
                M_PI/10, M_PI/6, M_PI/6,
                1.0,   1.0,   1.0,
                1.0,   1.0,   1.0,
                1.0,   1.0,   1.0,
                1.0,   1.0,   1.0;
        actionMean_<< 0.0, 0.0, 0.0, 0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0,
                0.0,     0.0,     0.0;
        //FHs_.resize(4);
        ex_encoder_inputs.resize(1);
        belief_encoder_inputs.resize(3);
        actor_inputs.resize(3);
        belief_decoder_inputs.resize(2);
        o_pt_now.setZero(154);
        o_et.setZero(208);
        h_t.setZero();
        E_mean = readMatrix("mean.csv");
        E_var = readMatrix("var.csv");
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            ex_encoder = torch::jit::load("ex_encoder.pt");
            belief_encoder = torch::jit::load("belief_encoder.pt");
            actor = torch::jit::load("actor.pt");
            belief_decoder = torch::jit::load("belief_decoder.pt");
          }
          catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
          }
        for(int i = 0;i<4;i++)
        {
//                    Joint_action_params_swing[i][0].kp_ = 100.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][0].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][0].kff_ = 50.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][1].kp_ = 100.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][1].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][1].kff_ = 70.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][2].kp_ = 50.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][2].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][2].kff_ = 150.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    //0602
//                    Joint_action_params_swing[i][0].kp_ = 100.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][0].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][0].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][1].kp_ = 100.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][1].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][1].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][2].kp_ = 50.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_swing[i][2].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_swing[i][2].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;

//                    Joint_action_params_stand[i][0].kp_ = 30.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_stand[i][0].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_stand[i][0].kff_ = 50.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_stand[i][1].kp_ = 30.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_stand[i][1].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_stand[i][1].kff_ = 70.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_stand[i][2].kp_ = 30.0;//300.0;//150.0;//150.0f;
//                    Joint_action_params_stand[i][2].kd_ = 0.6;//2.5;//1.3f;//1.3;//3.0;//1.3;
//                    Joint_action_params_stand[i][2].kff_ = 150.0;//2.5;//1.3f;//1.3;//3.0;//1.3;


            //0607 no switch
            Joint_action_params_swing[i][0].kp_ = 100.0;//300.0;//150.0;//150.0f;
            Joint_action_params_swing[i][0].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_swing[i][0].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_swing[i][1].kp_ = 100.0;//300.0;//150.0;//150.0f;
            Joint_action_params_swing[i][1].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_swing[i][1].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_swing[i][2].kp_ = 50.0;//300.0;//150.0;//150.0f;
            Joint_action_params_swing[i][2].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_swing[i][2].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;

            Joint_action_params_stand[i][0].kp_ = 100.0;//300.0;//150.0;//150.0f;
            Joint_action_params_stand[i][0].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_stand[i][0].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_stand[i][1].kp_ = 100.0;//300.0;//150.0;//150.0f;
            Joint_action_params_stand[i][1].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_stand[i][1].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_stand[i][2].kp_ = 50.0;//300.0;//150.0;//150.0f;
            Joint_action_params_stand[i][2].kd_ = 1.3;//2.5;//1.3f;//1.3;//3.0;//1.3;
            Joint_action_params_stand[i][2].kff_ = 0.0;//2.5;//1.3f;//1.3;//3.0;//1.3;
        }

        train_first_run = 0;
        input.ocu_xyz[0] = 0;
        input.ocu_xyz[1] = 0;
        input.ocu_xyz[2] = 0;
        for(int i = 0; i < 3; i++)
        {

            input.BodyOrientation[i] = quaternionToRotationMatrix(_sharedMemory().simToRobot.cheaterState.orientation).row(2)[i];
            input.BodyAngularVel[i] = _sharedMemory().simToRobot.cheaterState.omegaBody[i];
        }
        for(int i = 0; i < 4; i++)
        {
            int k = 0;
            switch (i) {
            case 0:
                k=1;
                break;
            case 1:
                k=0;
                break;
            case 2:
                k=3;
                break;
            case 3:
                k=2;
                break;
            }
            for(int j = 0; j < 3; j++)
            {
                if(j == 0)
                {
                    input.jointPos[i*3+j] = _spiData.q_abad[k];
                    input.jointVel[i*3+j] = _spiData.qd_abad[k];
                }
                else if(j == 1)
                {
                    input.jointPos[i*3+j] = -_spiData.q_hip[k];
                    input.jointVel[i*3+j] = -_spiData.qd_hip[k];
                }
                else if(j == 2)
                {
                    input.jointPos[i*3+j] = -_spiData.q_knee[k];
                    input.jointVel[i*3+j] = -_spiData.qd_knee[k];
                }

                input.jointPosLast2[i*3+j] = input.jointPos[i*3+j];
                input.jointPosLast1[i*3+j] = input.jointPosLast2[i*3+j];
                input.jointPosLast0[i*3+j] = input.jointPosLast1[i*3+j];

                input.jointVelLast1[i*3+j] = input.jointVel[i*3+j];
                input.jointVelLast0[i*3+j] = input.jointVelLast1[i*3+j];


                input.jointPos_r_Last2[i*3+j] = input.jointPos[i*3+j];
                input.jointPos_r_Last1[i*3+j] = input.jointPos_r_Last2[i*3+j];

                input.baseFreq_ = 1.3 * freqScale_;

                input.jointff_r_Last1[i*3+j] = 0;
                input.jointff_r_Last2[i*3+j] = 0;
            }
            input.piD_[i] = 0;
            pi_[i] = 0;
            input.cspi_[i*2] = sin(pi_[i]);
            input.cspi_[i*2 + 1] = cos(pi_[i]);

           // FHs_[i] = Eigen::Matrix<float, 52, 1>::Zero();
        }
        std::cout<<"#####$$$$$INIT$$$$$######"<<std::endl;
        float * pinput = (float *)&input;
        for(int i=0;i<154;i++)
        {
            o_pt_now[i] = pinput[i];
        }
        o_et.setZero();
        h_t = Eigen::Matrix<float, 2, 50>::Random();
        std::cout<<"#####154######"<<o_pt_now<<std::endl;
        //std::cout<<"#####h_t######"<<h_t<<std::endl;
    }
    else
    {
        if(tick == 20)
        {
            // 换格式 并赋值
            o_pt_now_normal = ((o_pt_now - E_mean.head(154)).array() / ((E_var.head(154).array()+0.00000001).sqrt())).cwiseMin(10).cwiseMax(-10);
            o_et_normal = ((o_et - E_mean.tail(208)).array() / ((E_var.tail(208).array()+0.00000001).sqrt())).cwiseMin(10).cwiseMax(-10);

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            ex_encoder_inputs[0] = torch::from_blob(o_et_normal.data(), {1,1,208},options);
            belief_decoder_inputs[0] = ex_encoder_inputs[0];
            if(tick < 1)
            {
                std::cout << "###########ex_encoder_inputs#############" << '\n';
                std::cout << ex_encoder_inputs[0] << std::endl;
            }

            belief_encoder_inputs[0] = torch::from_blob(o_pt_now_normal.data(), {1,1,154},options);
            belief_encoder_inputs[2] = torch::from_blob(h_t.data(), {2,1,50},options);



            actor_inputs[0] = belief_encoder_inputs[0];


            belief_encoder_inputs[1] = ex_encoder.forward(ex_encoder_inputs);
            auto belief_encoder_output = belief_encoder.forward(belief_encoder_inputs).toTuple();
            actor_inputs[1] = belief_encoder_output->elements()[0];
            actor_inputs[2] = belief_encoder_output->elements()[1];
            belief_decoder_inputs[1] = belief_encoder_output->elements()[2];
            belief_encoder_inputs[2] = belief_encoder_output->elements()[3];

            auto ation = actor.forward(actor_inputs);
            auto decoded_info = belief_decoder.forward(belief_decoder_inputs).toTuple();
            if(tick < 1)
            {
                std::cout << "###########belief_encoder_inputs0#############" << '\n';
                std::cout << belief_encoder_inputs[0] << '\n';
                std::cout << "###########belief_encoder_inputs1#############" << '\n';
                std::cout << belief_encoder_inputs[1] << '\n';
                std::cout << "############actor_inputs1############" << '\n';
                std::cout << actor_inputs[1] << '\n';
                std::cout << "############actor_inputs2############" << '\n';
                std::cout << actor_inputs[2] << '\n';
                std::cout << "############ation############" << '\n';
                std::cout << ation << '\n';
                std::cout << "############decoded_info1############" << '\n';
                std::cout << decoded_info->elements()[0] << '\n';
                std::cout << "############decoded_info2############" << '\n';
                std::cout << decoded_info->elements()[1] << '\n';
            }


            at::Tensor action = ation.toTensor();
            float * action_data = action.data_ptr<float>();
            for(int i = 0; i< 28 ;i++)
                action_t[i] = action_data[i];

            float* p_h_t = belief_encoder_output->elements()[3].toTensor().data_ptr<float>();
            for(int j = 0;j<2;j++)
                for(int i = 0; i< 50 ;i++)
                    h_t(j,i) = p_h_t[j*50+i];
            processAction();
            Update_Train_Input();
            clock_gettime(CLOCK_REALTIME, &time2);
            timesim = (float)(time2.tv_sec - time1.tv_sec)*1000.0 + (float)(time2.tv_nsec - time1.tv_nsec)/1000000.0 ;
            printf("lastPeriodTime: %f\n",timesim);
            tick = 0;
        }
        if(tick == 1)
        {
            clock_gettime(CLOCK_REALTIME, &time1);
        }
        tick++;
    }
    for(int i = 0;i<4;i++)
    {
        int k = 0;
        switch (i) {
        case 0:
            k=1;
            break;
        case 1:
            k=0;
            break;
        case 2:
            k=3;
            break;
        case 3:
            k=2;
            break;
        }
        _spiCommand.kp_abad[k] = Joint_action_params_swing[i][0].kp_;
        _spiCommand.kd_abad[k] = Joint_action_params_swing[i][0].kd_;

        _spiCommand.kp_hip[k] = Joint_action_params_swing[i][1].kp_;
        _spiCommand.kd_hip[k] = Joint_action_params_swing[i][1].kd_;

        _spiCommand.kp_knee[k] = Joint_action_params_swing[i][2].kp_;
        _spiCommand.kd_knee[k] = Joint_action_params_swing[i][2].kd_;

    //            gait_Cmd.q_des_abad[k] = pTarget_joint_[i*3+0];
    //            gait_Cmd.qd_des_abad[k] = 0;

    //            gait_Cmd.q_des_hip[k] = -pTarget_joint_[i*3+1];
    //            gait_Cmd.qd_des_hip[k] = 0;

    //            gait_Cmd.q_des_knee[k] = -pTarget_joint_[i*3+2];
    //            gait_Cmd.qd_des_knee[k] = 0;

    //            gait_Cmd.tau_abad_ff[k] = Joint_action_params_swing[i][0].kff_ * scaledAction_[16+i*3+0];
    //            gait_Cmd.tau_hip_ff[k] = -Joint_action_params_swing[i][1].kff_ * scaledAction_[16+i*3+1];
    //            gait_Cmd.tau_knee_ff[k] = -Joint_action_params_swing[i][2].kff_ * scaledAction_[16+i*3+2];

    //            //0607
        _spiCommand.q_des_abad[k] = pTarget_joint_[i*3+0] + scaledAction_[16+i*3+0];
        _spiCommand.qd_des_abad[k] = 0;

        _spiCommand.q_des_hip[k] = -(pTarget_joint_[i*3+1] + scaledAction_[16+i*3+1]);
        _spiCommand.qd_des_hip[k] = 0;

        _spiCommand.q_des_knee[k] = -(pTarget_joint_[i*3+2] + scaledAction_[16+i*3+2]);
        _spiCommand.qd_des_knee[k] = 0;

        _spiCommand.tau_abad_ff[k] = 0;
        _spiCommand.tau_hip_ff[k] = 0;
        _spiCommand.tau_knee_ff[k] = 0;

        _spiCommand.tau_hip_ff[k] =_spiCommand.kp_hip[k] *
                (_spiCommand.q_des_hip[k] - _sharedMemory().simToRobot.spiData.q_hip[k]) +
            _spiCommand.kd_hip[k] * (_spiCommand.qd_des_hip[k] -
                                       _sharedMemory().simToRobot.spiData.qd_hip[k]) +
            _spiCommand.tau_hip_ff[k];

        _spiCommand.tau_abad_ff[k] =_spiCommand.kp_abad[k] *
                (_spiCommand.q_des_abad[k] - _sharedMemory().simToRobot.spiData.q_abad[k]) +
            _spiCommand.kd_abad[k] * (_spiCommand.qd_des_abad[k] -
                                       _sharedMemory().simToRobot.spiData.qd_abad[k]) +
            _spiCommand.tau_abad_ff[k];

        _spiCommand.tau_knee_ff[k] =_spiCommand.kp_knee[k] *
                (_spiCommand.q_des_knee[k] - _sharedMemory().simToRobot.spiData.q_knee[k]) +
            _spiCommand.kd_knee[k] * (_spiCommand.qd_des_knee[k] -
                                       _sharedMemory().simToRobot.spiData.qd_knee[k]) +
            _spiCommand.tau_knee_ff[k];

        _spiCommand.kp_hip[k] = 0;
        _spiCommand.kd_hip[k] = 0;
        _spiCommand.kp_abad[k] = 0;
        _spiCommand.kd_abad[k] = 0;
        _spiCommand.kp_knee[k] = 0;
        _spiCommand.kd_knee[k] = 0;
    }

}
else
{
    train_first_run = 1;
}
#endif








#endif


  _highLevelIterations++;
}

void Simulation::buildLcmMessage() {
#if 0
  _simLCM.time = _currentSimTime;
  _simLCM.timesteps = _highLevelIterations;
  auto& state = _simulator->getState();
  auto& dstate = _simulator->getDState();

  Vec3<double> rpy = ori::quatToRPY(state.bodyOrientation);
  RotMat<double> Rbody = ori::quaternionToRotationMatrix(state.bodyOrientation);
  Vec3<double> omega = Rbody.transpose() * state.bodyVelocity.head<3>();
  Vec3<double> v = Rbody.transpose() * state.bodyVelocity.tail<3>();

  for (size_t i = 0; i < 4; i++) {
    _simLCM.quat[i] = state.bodyOrientation[i];
  }

  for (size_t i = 0; i < 3; i++) {
    _simLCM.vb[i] = state.bodyVelocity[i + 3];  // linear velocity in body frame
    _simLCM.rpy[i] = rpy[i];
    for (size_t j = 0; j < 3; j++) {
      _simLCM.R[i][j] = Rbody(i, j);
    }
    _simLCM.omegab[i] = state.bodyVelocity[i];
    _simLCM.omega[i] = omega[i];
    _simLCM.p[i] = state.bodyPosition[i];
    _simLCM.v[i] = v[i];
    _simLCM.vbd[i] = dstate.dBodyVelocity[i + 3];
  }

  for (size_t leg = 0; leg < 4; leg++) {
    for (size_t joint = 0; joint < 3; joint++) {
      _simLCM.q[leg][joint] = state.q[leg * 3 + joint];
      _simLCM.qd[leg][joint] = state.qd[leg * 3 + joint];
      _simLCM.qdd[leg][joint] = dstate.qdd[leg * 3 + joint];
      _simLCM.tau[leg][joint] = _tau[leg * 3 + joint];
      size_t gcID = _simulator->getModel()._footIndicesGC.at(leg);
      _simLCM.p_foot[leg][joint] = _simulator->getModel()._pGC.at(gcID)[joint];
      _simLCM.f_foot[leg][joint] = _simulator->getContactForce(gcID)[joint];
    }
  }
#endif //no use ---xp
}

/*!
 * Add an infinite collision plane to the simulator
 * @param mu          : friction of the plane
 * @param resti       : restitution coefficient
 * @param height      : height of plane
 * @param addToWindow : if true, also adds graphics for the plane
 */
void Simulation::addCollisionPlane(double mu, double resti, double height,
                                   double sizeX, double sizeY, double checkerX,
                                   double checkerY, bool addToWindow) {
  _simulator->addCollisionPlane(mu, resti, height);
  if (addToWindow && _window) {
    _window->lockGfxMutex();
    Checkerboard checker(sizeX, sizeY, checkerX, checkerY);

    size_t graphicsID = _window->_drawList.addCheckerboard(checker, true);
    _window->_drawList.buildDrawList();
    _window->_drawList.updateCheckerboard(height, graphicsID);
    _window->unlockGfxMutex();
  }
}

/*!
 * Add an box collision to the simulator
 * @param mu          : location of the box
 * @param resti       : restitution coefficient
 * @param depth       : depth (x) of box
 * @param width       : width (y) of box
 * @param height      : height (z) of box
 * @param pos         : position of box
 * @param ori         : orientation of box
 * @param addToWindow : if true, also adds graphics for the plane
 */
void Simulation::addCollisionBox(double mu, double resti, double depth,
                                 double width, double height,
                                 const Vec3<double>& pos,
                                 const Mat3<double>& ori, bool addToWindow,
                                 bool transparent) {
  _simulator->addCollisionBox(mu, resti, depth, width, height, pos, ori);
  if (addToWindow && _window) {
    _window->lockGfxMutex();
    _window->_drawList.addBox(depth, width, height, pos, ori, transparent);
    _window->unlockGfxMutex();
  }
}

void Simulation::addCollisionMesh(double mu, double resti, double grid_size,
                                  const Vec3<double>& left_corner_loc,
                                  const DMat<double>& height_map,
                                  bool addToWindow, bool transparent) {
  _simulator->addCollisionMesh(mu, resti, grid_size, left_corner_loc,
                               height_map);
  if (addToWindow && _window) {
    _window->lockGfxMutex();
    _window->_drawList.addMesh(grid_size, left_corner_loc, height_map,
                               transparent);
    _window->unlockGfxMutex();
  }
}

/*!
 * Runs the simulator in the current thread until the _running variable is set
 * to false. Updates graphics at 60 fps if desired. Runs simulation at the
 * desired speed
 * @param dt
 */
void *func(void* p);
void Simulation::runAtSpeed(std::function<void(std::string)> errorCallback, bool graphics) {
  _errorCallback = errorCallback;
  firstRun();  // load the control parameters

  // if we requested to stop, stop.
  if (_wantStop) return;
  assert(!_running);
  _running = true;
  pthread_create(&sim_g, NULL, &func, this);
  Timer frameTimer;
  Timer freeRunTimer;
  u64 desiredSteps = 0;
  u64 steps = 0;

  double frameTime = 1. / 60.;
  double lastSimTime = 0;
  en_graphics = graphics;

  printf(
      "[Simulator] Starting run loop (dt %f, dt-low-level %f, dt-high-level %f "
      "speed %f graphics %d)...\n",
      _simParams.dynamics_dt, _simParams.low_level_dt, _simParams.high_level_dt,
      _simParams.simulation_speed, graphics);

  int seconds = (int)_simParams.dynamics_dt;
  int nanoseconds = (int)(1e9 * std::fmod(_simParams.dynamics_dt, 1.f));
  //printf("nanoseconds = %d\n", nanoseconds);

//  unsigned long long missed = 0;
//  auto timerFd = timerfd_create(CLOCK_REALTIME, 0);
//  itimerspec timerSpec;
//  timerSpec.it_interval.tv_sec = seconds;
//  timerSpec.it_value.tv_sec = seconds;
//  timerSpec.it_value.tv_nsec = nanoseconds;
//  timerSpec.it_interval.tv_nsec = nanoseconds;
//  timerfd_settime(timerFd, 0, &timerSpec, nullptr);



  while (_running) {
    struct timespec time1 = {0, 0};
    struct timespec time2 = {0, 0};
    double dt = _simParams.dynamics_dt;
    double dtLowLevelControl = _simParams.low_level_dt;
    double dtHighLevelControl = _simParams.high_level_dt;
    _desiredSimSpeed = (_window && _window->wantTurbo()) ? 100.f : _simParams.simulation_speed;
    if(_window && _window->wantSloMo()) {
      _desiredSimSpeed /= 10.;
    }

    //_desiredSimSpeed = 1.0f  from params file
//    printf(
//        "[Simulator] Starting run loop (dt %f, dt-low-level %f, dt-high-level %f "
//        "speed %f graphics %d)...\n",
//        _simParams.dynamics_dt, _simParams.low_level_dt, _simParams.high_level_dt,
//        _simParams.simulation_speed, graphics);


    u64 nStepsPerFrame = (u64)(((1. / 60.) / dt) * _desiredSimSpeed);
    if ((!_window->IsPaused() && steps < desiredSteps)||1) {                //qiu
      clock_gettime(CLOCK_REALTIME, &time1);
      _simParams.lockMutex();
      step(dt, dtLowLevelControl, dtHighLevelControl);//core --- update simulator ---xp
      _simParams.unlockMutex();
//      int m = read(timerFd, &missed, sizeof(missed));
//      (void)m;
      clock_gettime(CLOCK_REALTIME, &time2);
      float timesim = (float)(time2.tv_sec - time1.tv_sec)*1000.0 + (float)(time2.tv_nsec - time1.tv_nsec)/1000000.0 ;
      while(timesim - dt * 1000.0<=0.000001)
      {
          clock_gettime(CLOCK_REALTIME, &time2);
          timesim = (float)(time2.tv_sec - time1.tv_sec)*1000.0 + (float)(time2.tv_nsec - time1.tv_nsec)/1000000.0 ;
      }

      if(timesim - dt * 1000.0 > 0.1)
      {
          static unsigned long over_step = 0;
          over_step++;
          cout <<"["<< steps << "] sim overtime: " << timesim << "ms over_persent "<<100.0 * (float)over_step/(float)steps << "%"<< endl;
      }
      steps++;
    } else {
      double timeRemaining = frameTime - frameTimer.getSeconds();
      if (timeRemaining > 0) {
        usleep((u32)(timeRemaining * 1e6));
      }
    }
//    if (frameTimer.getSeconds() > frameTime) {
//      double realElapsedTime = frameTimer.getSeconds();
//      frameTimer.start();
//      if (graphics && _window) {
////        double simRate = (_currentSimTime - lastSimTime) / realElapsedTime;
////        lastSimTime = _currentSimTime;
////        sprintf(_window->infoString,
////                "[Simulation Run %5.2fx]\n"
////                "real-time:  %8.3f\n"
////                "sim-time:   %8.3f\n"
////                "rate:       %8.3f\n",
////                _desiredSimSpeed, freeRunTimer.getSeconds(), _currentSimTime,
////                simRate);
//        updateGraphics();
//      }
//      if (!_window->IsPaused() && (desiredSteps - steps) < nStepsPerFrame)
//        desiredSteps += nStepsPerFrame;
//    }
  }
}

void *func(void* p)
{
    Simulation *  sim = (Simulation *)p;
    Timer frameTimer;
    double frameTime = 1. / 60.;
    while(sim->isRunning())
    {
        if (frameTimer.getSeconds() > frameTime) {
          double realElapsedTime = frameTimer.getSeconds();
          frameTimer.start();
          if (sim->isGraphicsEnable() && sim->getWindow()) {
    //        double simRate = (_currentSimTime - lastSimTime) / realElapsedTime;
    //        lastSimTime = _currentSimTime;
    //        sprintf(_window->infoString,
    //                "[Simulation Run %5.2fx]\n"
    //                "real-time:  %8.3f\n"
    //                "sim-time:   %8.3f\n"
    //                "rate:       %8.3f\n",
    //                _desiredSimSpeed, freeRunTimer.getSeconds(), _currentSimTime,
    //                simRate);
    //          printf("###########Graphics Update: %f s\n",realElapsedTime);
            sim->updateGraphics();
          }
       }
    }
}


void Simulation::loadTerrainFile(const std::string& terrainFileName,
                                 bool addGraphics) {
  printf("load terrain %s\n", terrainFileName.c_str());
  xpYaml paramHandler(terrainFileName);

  if (!paramHandler.fileOpenedSuccessfully()) {
    printf("[ERROR] could not open yaml file for terrain\n");
    throw std::runtime_error("yaml bad");
  }

  std::vector<std::string> keys = paramHandler.getKeys();

  for (auto& key : keys) {
    auto load = [&](double& val, const std::string& name) {
      if (!paramHandler.getValue<double>(key, name, val))
        throw std::runtime_error("terrain read bad: " + key + " " + name);
    };

    auto loadVec = [&](double& val, const std::string& name, size_t idx) {
      std::vector<double> v;
      if (!paramHandler.getVector<double>(key, name, v))
        throw std::runtime_error("terrain read bad: " + key + " " + name);
      val = v.at(idx);
    };

    auto loadArray = [&](double* val, const std::string& name, size_t idx) {
      std::vector<double> v;
      if (!paramHandler.getVector<double>(key, name, v))
        throw std::runtime_error("terrain read bad: " + key + " " + name);
      assert(v.size() == idx);
      for (size_t i = 0; i < idx; i++) val[i] = v[i];
    };

    printf("terrain element %s\n", key.c_str());
    std::string typeName;
    paramHandler.getString(key, "type", typeName);
    if (typeName == "infinite-plane") {
      double mu, resti, height, gfxX, gfxY, checkerX, checkerY;
      load(mu, "mu");
      load(resti, "restitution");
      load(height, "height");
      loadVec(gfxX, "graphicsSize", 0);
      loadVec(gfxY, "graphicsSize", 1);
      loadVec(checkerX, "checkers", 0);
      loadVec(checkerY, "checkers", 1);
      addCollisionPlane(mu, resti, height, gfxX, gfxY, checkerX, checkerY,
                        addGraphics);
    } else if (typeName == "box") {
      double mu, resti, depth, width, height, transparent;
      double pos[3];
      double ori[3];
      load(mu, "mu");
      load(resti, "restitution");
      load(depth, "depth");
      load(width, "width");
      load(height, "height");
      loadArray(pos, "position", 3);
      loadArray(ori, "orientation", 3);
      load(transparent, "transparent");

      Mat3<double> R_box = ori::rpyToRotMat(Vec3<double>(ori));
      R_box.transposeInPlace();  // collisionBox uses "rotation" matrix instead
                                 // of "transformation"
      addCollisionBox(mu, resti, depth, width, height, Vec3<double>(pos), R_box,
                      addGraphics, transparent != 0.);
    } else if (typeName == "stairs") {
      double mu, resti, rise, run, stepsDouble, width, transparent;
      double pos[3];
      double ori[3];
      load(mu, "mu");
      load(resti, "restitution");
      load(rise, "rise");
      load(width, "width");
      load(run, "run");
      load(stepsDouble, "steps");
      loadArray(pos, "position", 3);
      loadArray(ori, "orientation", 3);
      load(transparent, "transparent");

      Mat3<double> R = ori::rpyToRotMat(Vec3<double>(ori));
      Vec3<double> pOff(pos);
      R.transposeInPlace();  // "graphics" rotation matrix

      size_t steps = (size_t)stepsDouble;

      double heightOffset = rise / 2;
      double runOffset = run / 2;
      for (size_t step = 0; step < steps; step++) {
        Vec3<double> p(runOffset, 0, heightOffset);
        p = R * p + pOff;

        addCollisionBox(mu, resti, run, width, heightOffset * 2, p, R,
                        addGraphics, transparent != 0.);

        heightOffset += rise / 2;
        runOffset += run;
      }
    } else if (typeName == "mesh") {
      double mu, resti, transparent, grid;
      Vec3<double> left_corner;
      std::vector<std::vector<double> > height_map_2d;
      load(mu, "mu");
      load(resti, "restitution");
      load(transparent, "transparent");
      load(grid, "grid");
      loadVec(left_corner[0], "left_corner_loc", 0);
      loadVec(left_corner[1], "left_corner_loc", 1);
      loadVec(left_corner[2], "left_corner_loc", 2);

      int x_len(0);
      int y_len(0);
      bool file_input(false);
      paramHandler.getBoolean(key, "heightmap_file", file_input);
      if (file_input) {
        // Read from text file
        std::string file_name;
        paramHandler.getString(key, "heightmap_file_name", file_name);
        std::ifstream f_height;
        f_height.open(THIS_COM "/config/" + file_name);
        if (!f_height.good()) {
          std::cout << "file reading error: "
                    << THIS_COM "../config/" + file_name << std::endl;
        }
        int i(0);
        int j(0);
        double tmp;

        std::string line;
        std::vector<double> height_map_vec;
        while (getline(f_height, line)) {
          std::istringstream iss(line);
          j = 0;
          while (iss >> tmp) {
            height_map_vec.push_back(tmp);
            ++j;
          }
          y_len = j;
          height_map_2d.push_back(height_map_vec);
          height_map_vec.clear();
          // printf("y len: %d\n", y_len);
          ++i;
        }
        x_len = i;

      } else {
        paramHandler.get2DArray(key, "height_map", height_map_2d);
        x_len = height_map_2d.size();
        y_len = height_map_2d[0].size();
        // printf("x, y len: %d, %d\n", x_len, y_len);
      }

      DMat<double> height_map(x_len, y_len);
      for (int i(0); i < x_len; ++i) {
        for (int j(0); j < y_len; ++j) {
          height_map(i, j) = height_map_2d[i][j];
          // printf("height (%d, %d) : %f\n", i, j, height_map(i,j) );
        }
      }
      addCollisionMesh(mu, resti, grid, left_corner, height_map, addGraphics,
                       transparent != 0.);

    } else {
      throw std::runtime_error("unknown terrain " + typeName);
    }
  }
}

void Simulation::updateGraphics() {
  _robotControllerState.bodyOrientation =
      _sharedMemory().robotToSim.mainCheetahVisualization.quat.cast<double>();
  _robotControllerState.bodyPosition =
      _sharedMemory().robotToSim.mainCheetahVisualization.p.cast<double>();
  for (int i = 0; i < 12; i++)
    _robotControllerState.q[i] =
        _sharedMemory().robotToSim.mainCheetahVisualization.q[i];
  //_robotDataSimulator->setState(_robotControllerState);
  //_robotDataSimulator->forwardKinematics();  // calc all body positions
  _window->_drawList.updateRobotFromModel(*_simulator, _simRobotID, true);
//  _window->_drawList.updateRobotFromModel(*_robotDataSimulator,
//                                          _controllerRobotID, false);
  //_window->_drawList.updateAdditionalInfo(*_simulator);
  _window->update();
}


