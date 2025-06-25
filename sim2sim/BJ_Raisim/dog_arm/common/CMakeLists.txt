# Source/Include files
include_directories(".")
include_directories("include/" )                 # common includes
include_directories("include/comm/")
include_directories("include/utilities/")
include_directories("/usr/include/eigen3/")
include_directories("${PROJECT_SOURCE_DIR}/mid_level/include/")
include_directories("${PROJECT_SOURCE_DIR}/third_party/qpOASES/include/")
include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/")
include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/dynacore_yaml-cpp/")
include_directories("${PROJECT_SOURCE_DIR}/api_sdk/include/")
include_directories("${PROJECT_SOURCE_DIR}/common/comm")
include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim")
file(GLOB_RECURSE sources "src/*.cpp")                   # common library cpp files
set(Qt5Core_DIR /home/pcpc/Qt/5.15.0/gcc_64/lib/cmake/Qt5Core)
find_package(Qt5Core CONFIG REQUIRED)

add_library(xp_common_lib SHARED ${sources})       # produce a library used by others
target_link_libraries(xp_common_lib pthread rt qpOASES ParamYaml Qt5::Core)
if(USING_CASADI_ENABLE)
    target_link_libraries(xp_common_lib casadi)
endif(USING_CASADI_ENABLE)
