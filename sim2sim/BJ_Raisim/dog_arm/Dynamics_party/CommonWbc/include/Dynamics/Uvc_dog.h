
#ifndef PROJECT_MINICHEETAH_H
#define PROJECT_MINICHEETAH_H

#include "FloatingBaseModel.h"
#include "Quadruped.h"
#define _USE_URDF 0
template <typename T>
Quadruped<T> buildUvcDog() {

  Quadruped<T> UvcDog;
#if 1//!_USE_SIM
#if !_USE_URDF

  UvcDog._robotType = RobotType::Uvc_dog_proto;
#if _USE_WJ
  UvcDog._bodyMass = 44.08+60;//40;//20;//61.66;//61.66;//40.0; 3.0f
#else
  UvcDog._bodyMass = 44.08;//40;//20;//61.66;//61.66;//40.0; 3.0f
#endif

   UvcDog._bodyLength = 0.66;
   UvcDog._bodyWidth = 0.26;
   UvcDog._bodyHeight = 0.15;
   UvcDog._abadGearRatio = 16.227;
   UvcDog._hipGearRatio = 16.227;
   UvcDog._kneeGearRatio = 16.227;
   UvcDog._abadLinkLength = 0.10575;
   UvcDog._hipLinkLength = 0.34;

   UvcDog._kneeLinkY_offset = 0.0;
   UvcDog._kneeLinkLength = sqrt(0.37151*0.37151+0.01478*0.01478);
   UvcDog._maxLegLength = UvcDog._kneeLinkLength + UvcDog._hipLinkLength;


   UvcDog._motorTauMax = 600.0;//200.0;
   UvcDog._batteryV = 48;//48;
   UvcDog._motorKT = 0.05;//0.281;//0.05f;//.05;  // this is flux linkage * pole pairs
   UvcDog._motorR = 0.173f;//0.4554;//0.173f;//0.173;//0.4554;//0.4554;//0.173;//0.4554;//0.173;//0.4554;//0.173;
   UvcDog._jointDamping = .01;
   UvcDog._jointDryFriction = .2;//123


   // rotor inertia if the rotor is oriented so it spins around the z-axis
   Mat3<T> rotorRotationalInertiaZ;
   rotorRotationalInertiaZ << 27.8, 0, 0, 0, 27.8, 0, 0, 0, 43.5;
   rotorRotationalInertiaZ = 1e-6 * rotorRotationalInertiaZ;

   Mat3<T> RY = coordinateRotation<T>(CoordinateAxis::Y, M_PI / 2);
   Mat3<T> RX = coordinateRotation<T>(CoordinateAxis::X, M_PI / 2);
   Mat3<T> rotorRotationalInertiaX =
       RY * rotorRotationalInertiaZ * RY.transpose();
   Mat3<T> rotorRotationalInertiaY =
       RX * rotorRotationalInertiaZ * RX.transpose();

   // spatial inertias
   Mat3<T> abadRotationalInertia;
   abadRotationalInertia << 381, 58, 0.45, 58, 560, 0.95, 0.45, 0.95, 444;
   abadRotationalInertia = abadRotationalInertia * 1e-6;
   Vec3<T> abadCOM(0, 0.05, 0);  // LEFT
   SpatialInertia<T> abadInertia(0.54, abadCOM, abadRotationalInertia);

   Mat3<T> hipRotationalInertia;
   hipRotationalInertia << 14260, 0, 0, 0, 14697, 0, 0, 0, 673;
   hipRotationalInertia = hipRotationalInertia * 1e-6;
   Vec3<T> hipCOM(0, 0.0, -0.15);
   SpatialInertia<T> hipInertia(0.84, hipCOM, hipRotationalInertia);

   Mat3<T> kneeRotationalInertia, kneeRotationalInertiaRotated;
   kneeRotationalInertiaRotated << 174, 0, 0, 0, 8752, 0, 0, 0, 8825;
   kneeRotationalInertiaRotated = kneeRotationalInertiaRotated * 1e-6;
   kneeRotationalInertia = RY * kneeRotationalInertiaRotated * RY.transpose();
   Vec3<T> kneeCOM(0, 0, -0.159);
   SpatialInertia<T> kneeInertia(0.50, kneeCOM, kneeRotationalInertia);



   Vec3<T> rotorCOM(0, 0, 0);
   SpatialInertia<T> rotorInertiaX(2.3, rotorCOM, rotorRotationalInertiaX);
   SpatialInertia<T> rotorInertiaY(2.3, rotorCOM, rotorRotationalInertiaY);

   Mat3<T> bodyRotationalInertia;
   bodyRotationalInertia << 336798, 0, 0, 0, 1425190, 0, 0, 0, 1565647;
   bodyRotationalInertia = bodyRotationalInertia * 1e-6;
   //std::cout<<bodyRotationalInertia<<endl;

   Vec3<T> bodyCOM(0, 0, 0);
   SpatialInertia<T> bodyInertia(UvcDog._bodyMass, bodyCOM,
                                 bodyRotationalInertia);

   UvcDog._abadInertia = abadInertia;
   UvcDog._hipInertia = hipInertia;
   UvcDog._kneeInertia = kneeInertia;
   UvcDog._abadRotorInertia = rotorInertiaX;
   UvcDog._hipRotorInertia = rotorInertiaY;
   UvcDog._kneeRotorInertia = rotorInertiaY;
   UvcDog._bodyInertia = bodyInertia;

   // locations
   UvcDog._abadRotorLocation = Vec3<T>(0.21, 0.13, 0);
   UvcDog._abadLocation =
       Vec3<T>(UvcDog._bodyLength, UvcDog._bodyWidth, 0) * 0.5;
   UvcDog._hipLocation = Vec3<T>(0, UvcDog._abadLinkLength, 0);
   UvcDog._hipRotorLocation = Vec3<T>(0, 0, 0);
   UvcDog._kneeLocation = Vec3<T>(0, 0, -UvcDog._hipLinkLength);
   UvcDog._kneeRotorLocation = Vec3<T>(0, 0, 0);

#ifdef _WHEEL_EN
   UvcDog._wheelLinkLength = 0;
   UvcDog._wheelGearRatio = 16.227;

   Mat3<T> wheelRotationalInertia;
   wheelRotationalInertia << 150, 0, 0, 0, 150, 0, 0, 0, 200;
   wheelRotationalInertia = wheelRotationalInertia * 1e-6;

   Vec3<T> wheelCOM(0, 0, 0);
   SpatialInertia<T> wheelInertia(0.50, wheelCOM, wheelRotationalInertia);
   UvcDog._wheelInertia = wheelInertia;
   UvcDog._wheelLocation = Vec3<T>(0, 0, -UvcDog._kneeLinkLength);
   UvcDog._wheelRotorLocation = Vec3<T>(0, 0, 0);
#endif
#else
  UvcDog._robotType = RobotType::Uvc_dog_proto;

   UvcDog._bodyMass = 32.145;//61.66;//61.66;//40.0; 3.0f
   UvcDog._bodyLength = 0.66;
   UvcDog._bodyWidth = 0.26;
   UvcDog._bodyHeight = 0.15;
   UvcDog._abadGearRatio = 16.227;
   UvcDog._hipGearRatio = 16.227;
   UvcDog._kneeGearRatio = 16.227;
   UvcDog._abadLinkLength = 0.10575;
   UvcDog._hipLinkLength = 0.34;

   UvcDog._kneeLinkY_offset = 0.0;
   UvcDog._kneeLinkLength = sqrt(0.37151*0.37151+0.01478*0.01478);
   UvcDog._maxLegLength = UvcDog._kneeLinkLength + UvcDog._hipLinkLength;


   UvcDog._motorTauMax = 12.5;//600.0;//200.0;
   UvcDog._batteryV = 48;//48;
   UvcDog._motorKT = 0.281;//0.05;//0.281;//0.05f;//.05;  // this is flux linkage * pole pairs
   UvcDog._motorR = 0.4554;//0.173f;//0.4554;//0.173f;//0.173;//0.4554;//0.4554;//0.173;//0.4554;//0.173;//0.4554;//0.173;
   UvcDog._jointDamping = .01;
   UvcDog._jointDryFriction = .2;//123


   // rotor inertia if the rotor is oriented so it spins around the z-axis
   Mat3<T> rotorRotationalInertiaZ;
   rotorRotationalInertiaZ << 0, 0, 0, 0, 0, 0, 0, 0, 188;
   rotorRotationalInertiaZ = 1e-6 * rotorRotationalInertiaZ;

   Mat3<T> RY = coordinateRotation<T>(CoordinateAxis::Y, M_PI / 2);
   Mat3<T> RX = coordinateRotation<T>(CoordinateAxis::X, M_PI / 2);
   Mat3<T> rotorRotationalInertiaX =
       RY * rotorRotationalInertiaZ * RY.transpose();
   Mat3<T> rotorRotationalInertiaY =
       RX * rotorRotationalInertiaZ * RX.transpose();

   // spatial inertias
   Mat3<T> abadRotationalInertia;
   abadRotationalInertia << 3194.584, 0, 0, 0, 4760.612, 0, 0, 0, 3606.273;
   abadRotationalInertia = abadRotationalInertia * 1e-6;
   Vec3<T> abadCOM(0.080, -0.023, 0);  // LEFT
   SpatialInertia<T> abadInertia( 2.410, abadCOM, abadRotationalInertia);

   Mat3<T> hipRotationalInertia;
   hipRotationalInertia <<  22276.074, 0, 0, 0, 22458.702, 0, 0, 0, 4295.608;
   hipRotationalInertia = hipRotationalInertia * 1e-6;
   Vec3<T> hipCOM(0, -0.036, -0.0386);
   SpatialInertia<T> hipInertia(2.821, hipCOM, hipRotationalInertia);

   Mat3<T> kneeRotationalInertia, kneeRotationalInertiaRotated;
   kneeRotationalInertia << 5908.688, 0, 0, 0, 5938.282, 0, 0, 0, 123.318;
   kneeRotationalInertia = kneeRotationalInertia * 1e-6;
   Vec3<T> kneeCOM(0, 0, -0.199);
   SpatialInertia<T> kneeInertia(0.470, kneeCOM, kneeRotationalInertia);



   Vec3<T> rotorCOM(0, 0, 0);
   SpatialInertia<T> rotorInertiaX(0.000, rotorCOM, rotorRotationalInertiaX);
   std::cout << "rotorX" << rotorRotationalInertiaX << std::endl;
   SpatialInertia<T> rotorInertiaY(0.000, rotorCOM, rotorRotationalInertiaY);
   std::cout << "rotorY" << rotorRotationalInertiaY << std::endl;

   Mat3<T> bodyRotationalInertia;
   bodyRotationalInertia << 210689.441, 0, 0, 0, 1441568.345, 0, 0, 0, 1596230.463;
   bodyRotationalInertia = bodyRotationalInertia * 1e-6;
   //std::cout<<bodyRotationalInertia<<endl;

   Vec3<T> bodyCOM(0, 0, -0.0315);
   SpatialInertia<T> bodyInertia(UvcDog._bodyMass, bodyCOM,
                                 bodyRotationalInertia);

   UvcDog._abadInertia = abadInertia;
   UvcDog._hipInertia = hipInertia;
   UvcDog._kneeInertia = kneeInertia;
   UvcDog._abadRotorInertia = rotorInertiaX;
   UvcDog._hipRotorInertia = rotorInertiaY;
   UvcDog._kneeRotorInertia = rotorInertiaY;
   UvcDog._bodyInertia = bodyInertia;

   // locations
   UvcDog._abadRotorLocation = Vec3<T>(0.21, 0.13, 0);
   UvcDog._abadLocation =
       Vec3<T>(UvcDog._bodyLength, UvcDog._bodyWidth, 0) * 0.5;
   UvcDog._hipLocation = Vec3<T>(0, UvcDog._abadLinkLength, 0);
   UvcDog._hipRotorLocation = Vec3<T>(0, 0, 0);
   UvcDog._kneeLocation = Vec3<T>(0, 0, -UvcDog._hipLinkLength);
   UvcDog._kneeRotorLocation = Vec3<T>(0, 0, 0);
#endif
#ifdef _USE_ARM
for(int arm_link = 0; arm_link<_DOF_ARM; arm_link++)
{
    UvcDog._armGearRatio[arm_link] = 20;
}
//inertias
  Mat3<T> armBaseRotationalInertia;
  armBaseRotationalInertia << 24200.021, 0, 0, 0, 7641.403, 0, 0, 0, 2006.144;
  armBaseRotationalInertia = armBaseRotationalInertia * 1e-6;
  Vec3<T> armBaseCOM(0, 0, 0.06);
  SpatialInertia<T> armBaseInertia(1.334, armBaseCOM,
                                armBaseRotationalInertia);

  Mat3<T> armLink1RotationalInertia;
  armLink1RotationalInertia << 762.264, 0, 0, 0, 813.146, 0, 0, 0, 720.740;
  armLink1RotationalInertia = armLink1RotationalInertia * 1e-6;
  Vec3<T> armLink1COM(0, 0, 0.04);
  SpatialInertia<T> armLink1Inertia(0.887, armLink1COM,
                                armLink1RotationalInertia);
  Mat3<T> armLink2RotationalInertia;
  armLink2RotationalInertia << 1477.084, 0, 0, 0, 25157.058, 0, 0, 0, 24841.039;
  armLink2RotationalInertia = armLink2RotationalInertia * 1e-6;
  Vec3<T> armLink2COM(0.210, 0, 0);
  SpatialInertia<T> armLink2Inertia(1.662, armLink2COM,
                                armLink2RotationalInertia);
  Mat3<T> armLink3RotationalInertia;
  armLink3RotationalInertia << 1560.666, 0, 0, 0, 21242.251, 0, 0, 0, 21639.776;
  armLink3RotationalInertia = armLink3RotationalInertia * 1e-6;
  Vec3<T> armLink3COM(-0.139, 0, 0.0357);
  SpatialInertia<T> armLink3Inertia(1.254, armLink3COM,
                                armLink3RotationalInertia);
  Mat3<T> armLink4RotationalInertia;
  armLink4RotationalInertia << 173.374, 0, 0, 0, 171.762, 0, 0, 0, 155.067;
  armLink4RotationalInertia = armLink4RotationalInertia * 1e-6;
  Vec3<T> armLink4COM(0.00138, 0, 0.0357);
  SpatialInertia<T> armLink4Inertia(0.342, armLink4COM,
                                armLink4RotationalInertia);
  Mat3<T> armLink5RotationalInertia;
  armLink5RotationalInertia << 330.120, 0, 0, 0, 647.973, 0, 0, 0, 550.722;
  armLink5RotationalInertia = armLink5RotationalInertia * 1e-6;
  Vec3<T> armLink5COM(-0.0369, 0, 0);
  SpatialInertia<T> armLink5Inertia(0.488, armLink5COM,
                                armLink5RotationalInertia);
  Mat3<T> armLink6RotationalInertia;
  armLink6RotationalInertia << 33.427, 0, 0, 0, 33.427, 0, 0, 0, 65.926;
  armLink6RotationalInertia = armLink6RotationalInertia * 1e-6;
  Vec3<T> armLink6COM(0, 0, 0.004);
  SpatialInertia<T> armLink6Inertia(0.15, armLink6COM,
                                armLink6RotationalInertia);


  SpatialInertia<T> rotorInertia(0.05, rotorCOM, rotorRotationalInertiaZ*0.5);

  UvcDog._armInertia[0] = armLink1Inertia;
  UvcDog._armInertia[1] = armLink2Inertia;
  UvcDog._armInertia[2] = armLink3Inertia;
  UvcDog._armInertia[3] = armLink4Inertia;
  UvcDog._armInertia[4] = armLink5Inertia;
  UvcDog._armInertia[5] = armLink6Inertia;
  UvcDog._armRotorInertia[0] = rotorInertia;
  UvcDog._armRotorInertia[1] = rotorInertia;
  UvcDog._armRotorInertia[2] = rotorInertia;
  UvcDog._armRotorInertia[3] = rotorInertia;
  UvcDog._armRotorInertia[4] = rotorInertia;
  UvcDog._armRotorInertia[5] = rotorInertia;



// locations
  //base
//      UvcDog._armBaseLocation = Vec3<T>(UvcDog._bodyLength* 0.5, 0, 0.12);
//      UvcDog._armBaseRotorLocation = Vec3<T>(0, 0, 0);
//      UvcDog._armLink1Location = Vec3<T>(0, 0, 0.115);
//      UvcDog._armLink1RotorLocation = Vec3<T>(0, 0, 0);
//      UvcDog._armLink2Location = Vec3<T>(0, 0, 0.045);
//      UvcDog._armLink2RotorLocation = Vec3<T>(0, 0, 0);
//      UvcDog._armLink3Location = Vec3<T>(-0.310, 0, 0);
//      UvcDog._armLink3RotorLocation = Vec3<T>(0, 0, 0);
//      UvcDog._armLink4Location = Vec3<T>(0.32697, 0.071, 0);
//      UvcDog._armLink4RotorLocation = Vec3<T>(0, 0, 0);
//      UvcDog._armLink5Location = Vec3<T>(0, 0, 0);
//      UvcDog._armLink5RotorLocation = Vec3<T>(0, 0, 0);
//      UvcDog._armLink6Location = Vec3<T>(0, 0, 0);
//      UvcDog._armLink6RotorLocation = Vec3<T>(0, 0, 0);
  //no base
  UvcDog._armLocation[0] = Vec3<T>(UvcDog._bodyLength* 0.5, 0, 0.115+0.12);
  UvcDog._armRotorLocation[0] = Vec3<T>(0, 0, 0);
  UvcDog._armLocation[1] = Vec3<T>(0, 0, 0.045);
  UvcDog._armRotorLocation[1] = Vec3<T>(0, 0, 0);
  UvcDog._armLocation[2] = Vec3<T>(-0.310, 0, 0);
  UvcDog._armRotorLocation[2] = Vec3<T>(0, 0, 0);
  UvcDog._armLocation[3] = Vec3<T>(0.32697, 0, 0.071);
  UvcDog._armRotorLocation[3] = Vec3<T>(0, 0, 0);
  UvcDog._armLocation[4] = Vec3<T>(0.03958, 0.0, 0.0);
  UvcDog._armRotorLocation[4] = Vec3<T>(0, 0, 0);
  UvcDog._armLocation[5] = Vec3<T>(0.08, 0.005, 0);
  UvcDog._armRotorLocation[5] = Vec3<T>(0, 0, 0);
#endif
#ifdef _WHEEL_EN
   UvcDog._wheelLinkLength = 0;
   UvcDog._wheelGearRatio = 16.227;

   Mat3<T> wheelRotationalInertia;
   wheelRotationalInertia << 150, 0, 0, 0, 150, 0, 0, 0, 200;
   wheelRotationalInertia = wheelRotationalInertia * 1e-6;

   Vec3<T> wheelCOM(0, 0, 0);
   SpatialInertia<T> wheelInertia(0.50, wheelCOM, wheelRotationalInertia);
   UvcDog._wheelInertia = wheelInertia;
   UvcDog._wheelLocation = Vec3<T>(0, 0, -UvcDog._kneeLinkLength);
   UvcDog._wheelRotorLocation = Vec3<T>(0, 0, 0);
#endif
#else
  UvcDog._robotType = RobotType::Uvc_dog_proto;

  UvcDog._bodyMass = 60.0;//3.0;//61.66;//40.0; 3.0f
  UvcDog._bodyLength = 0.66;
  UvcDog._bodyWidth = 0.26;
  UvcDog._bodyHeight = 0.15;
  UvcDog._abadGearRatio = 16.227;
  UvcDog._hipGearRatio = 16.227;
  UvcDog._kneeGearRatio = 16.227;
  UvcDog._abadLinkLength = 0.10575;
  UvcDog._hipLinkLength = 0.34;

  UvcDog._kneeLinkY_offset = 0.0;
  UvcDog._kneeLinkLength = sqrt(0.37151*0.37151+0.01478*0.01478);
  UvcDog._maxLegLength = UvcDog._kneeLinkLength + UvcDog._hipLinkLength;


  UvcDog._motorTauMax =300.0;//200.0;
  UvcDog._batteryV = 48;
  UvcDog._motorKT = 0.05;//.05;  // this is flux linkage * pole pairs
  UvcDog._motorR = 0.173f;//0.4554;//0.173;
  UvcDog._jointDamping = .01;
  UvcDog._jointDryFriction = .2;


  // rotor inertia if the rotor is oriented so it spins around the z-axis
  Mat3<T> rotorRotationalInertiaZ;
  rotorRotationalInertiaZ << 33, 0, 0, 0, 33, 0, 0, 0, 63;
  rotorRotationalInertiaZ = 1e-6 * rotorRotationalInertiaZ;

  Mat3<T> RY = coordinateRotation<T>(CoordinateAxis::Y, M_PI / 2);
  Mat3<T> RX = coordinateRotation<T>(CoordinateAxis::X, M_PI / 2);
  Mat3<T> rotorRotationalInertiaX =
      RY * rotorRotationalInertiaZ * RY.transpose();
  Mat3<T> rotorRotationalInertiaY =
      RX * rotorRotationalInertiaZ * RX.transpose();

  // spatial inertias
  Mat3<T> abadRotationalInertia;
  abadRotationalInertia << 381, 58, 0.45, 58, 560, 0.95, 0.45, 0.95, 444;
  abadRotationalInertia = abadRotationalInertia * 1e-6;
  Vec3<T> abadCOM(0, 0.05, 0);  // LEFT
  SpatialInertia<T> abadInertia(0.54, abadCOM, abadRotationalInertia);

  Mat3<T> hipRotationalInertia;
  hipRotationalInertia << 14260, 0, 0, 0, 14697, 0, 0, 0, 673;
  hipRotationalInertia = hipRotationalInertia * 1e-6;
  Vec3<T> hipCOM(0, 0.0, -0.15);
  SpatialInertia<T> hipInertia(0.84, hipCOM, hipRotationalInertia);

  Mat3<T> kneeRotationalInertia, kneeRotationalInertiaRotated;
  kneeRotationalInertiaRotated << 174, 0, 0, 0, 8752, 0, 0, 0, 8825;
  kneeRotationalInertiaRotated = kneeRotationalInertiaRotated * 1e-6;
  kneeRotationalInertia = RY * kneeRotationalInertiaRotated * RY.transpose();
  Vec3<T> kneeCOM(0, 0, -0.159);
  SpatialInertia<T> kneeInertia(0.50, kneeCOM, kneeRotationalInertia);

  Vec3<T> rotorCOM(0, 0, 0);
  SpatialInertia<T> rotorInertiaX(0.055, rotorCOM, rotorRotationalInertiaX);
  SpatialInertia<T> rotorInertiaY(0.055, rotorCOM, rotorRotationalInertiaY);

  Mat3<T> bodyRotationalInertia;
  bodyRotationalInertia << 336798, 0, 0, 0, 1425190, 0, 0, 0, 1565647;
  bodyRotationalInertia = bodyRotationalInertia * 1e-6;
  Vec3<T> bodyCOM(0, 0, 0);
  SpatialInertia<T> bodyInertia(UvcDog._bodyMass, bodyCOM,
                                bodyRotationalInertia);

  UvcDog._abadInertia = abadInertia;
  UvcDog._hipInertia = hipInertia;
  UvcDog._kneeInertia = kneeInertia;
  UvcDog._abadRotorInertia = rotorInertiaX;
  UvcDog._hipRotorInertia = rotorInertiaY;
  UvcDog._kneeRotorInertia = rotorInertiaY;
  UvcDog._bodyInertia = bodyInertia;

  // locations
  UvcDog._abadRotorLocation = Vec3<T>(0.125, 0.049, 0);//Vec3<T>(0.21, 0.13, 0);
  UvcDog._abadLocation =
      Vec3<T>(UvcDog._bodyLength, UvcDog._bodyWidth, 0) * 0.5;
  UvcDog._hipLocation = Vec3<T>(0, UvcDog._abadLinkLength, 0);
  UvcDog._hipRotorLocation = Vec3<T>(0, 0, 0);
  UvcDog._kneeLocation = Vec3<T>(0, 0, -UvcDog._hipLinkLength);
  UvcDog._kneeRotorLocation = Vec3<T>(0, 0, 0);
#endif
  return UvcDog;
}

#endif  // PROJECT_MINICHEETAH_H
