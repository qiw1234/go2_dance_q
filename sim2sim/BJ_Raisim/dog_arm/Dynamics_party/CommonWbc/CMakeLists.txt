# Source/Include files
include_directories(".")
include_directories("include/" )

include_directories(".")
include_directories("include/" )                 # common includes
include_directories("${PROJECT_SOURCE_DIR}/common/include/")
include_directories("${PROJECT_SOURCE_DIR}/common/include/comm")
include_directories("${PROJECT_SOURCE_DIR}/common/include/utilities")
include_directories("/usr/include/eigen3")
include_directories(${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})


include_directories(${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE sources "src/*.cpp")                   # common library cpp files

# Library
#add_library(commonWBC_lib SHARED ${sources})       # produce a library used by sim/robot
#target_link_libraries(commonWBC)


