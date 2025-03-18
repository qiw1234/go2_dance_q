#cmake_minimum_required(VERSION 3.5)
#project(raisim_examples LANGUAGES CXX)

#==
# Dependencies
#==
set(raisim_DIR /home/pcpc/raisim/raisim_build/lib/cmake/raisim)
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})


file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/rsc" DESTINATION ${CMAKE_BINARY_DIR})

include_directories ( ${EIGEN3_INCLUDE_DIRS} )
include_directories("${PROJECT_SOURCE_DIR}/common/include/")
include_directories("${PROJECT_SOURCE_DIR}/sim/include/")
include_directories("${PROJECT_SOURCE_DIR}/api_sdk/include/")
include_directories("${PROJECT_SOURCE_DIR}/lower_level/include/")
include_directories("${PROJECT_SOURCE_DIR}/mid_level/include/")
include_directories("${PROJECT_SOURCE_DIR}/upper_level/include/")
include_directories("${PROJECT_SOURCE_DIR}/third_party/qpOASES/include/")
include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/")
include_directories("${PROJECT_SOURCE_DIR}/third_party/EphCan/include")
include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/dynacore_yaml-cpp/")

include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/ControlParameters")
include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/Utilities")
include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/SimUtilities")
include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/Collision")
include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim")



#==
# Build Targets
#==

function(create_executable app_name file_name)
    
    add_executable(${app_name} ${file_name})
    set_target_properties(${app_name} PROPERTIES MACOSX_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/mac/lib")
    if(WIN32)
        target_link_libraries(${app_name} PUBLIC raisim::raisim Ws2_32 Winmm)
        target_compile_options(${app_name} PRIVATE "/MP")
    else()
        target_link_libraries(${app_name} PUBLIC raisim::raisim pthread xp_sim_lib)
    endif()

    if(APPLE)
        execute_process(COMMAND sysctl -q hw.optional.arm64
                OUTPUT_VARIABLE _sysctl_stdout
                ERROR_VARIABLE _sysctl_stderr
                RESULT_VARIABLE _sysctl_result
                )
        if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64: 1")
            target_compile_options(${app_name} PRIVATE -mcpu=apple-m1)
        endif()
    endif()

    target_include_directories(${app_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    #SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

#create_executable(panda5 src/panda5.cpp)
#SET(EXECUTABLE_OUTPUT_PATH "../${CMAKE_CURRENT_SOURCE_DIR}")
#file(GLOB_RECURSE sources
#"src/panda5.cpp")
#aux_source_directory(. SRC_LIST)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
if(USE_ARX_ARM)
create_executable(dog_sim2 src/panda5.cpp)
create_executable(dog_sim_arm_6dof src/panda5_6dof.cpp)
create_executable(go2_sim src/go2.cpp)
endif(USE_ARX_ARM)
#add_executable(dog_sim src/panda5.cpp)

#target_link_libraries(dog_sim ${libsources})
#target_link_libraries(dog_sim ${libsources_api})

