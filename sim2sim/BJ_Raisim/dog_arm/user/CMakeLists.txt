
add_subdirectory(DogActuator)
#add_subdirectory(ArmRaisim)
if(USING_RAISIM_ENABLE)
    add_subdirectory(DogSimRaisim)
else(USING_RAISIM_ENABLE)
    add_subdirectory(DogSim)
endif(USING_RAISIM_ENABLE)
if(TWO_RAISIMDOG_ENABLE)
    add_subdirectory(DogSimRaisim2)
endif(TWO_RAISIMDOG_ENABLE)

