add_subdirectory(CommonWbc)

#include_directories(".")
#include_directories("./")
#include_directories("./")
#include_directories("${PROJECT_SOURCE_DIR}/wbc/CommonWbc/include")
#include_directories("${PROJECT_SOURCE_DIR}/wbc")
#include_directories("${PROJECT_SOURCE_DIR}/wbc/WBC_Ctrl")

#include_directories(${CMAKE_BINARY_DIR})
#include_directories(${PROJECT_SOURCE_DIR})

##file(GLOB_RECURSE sources
##"./*.cpp")

##add_library(WBC_ctrl SHARED ${sources})

#include_directories(".")
#include_directories("include/" )                 # common includes
#include_directories("${PROJECT_SOURCE_DIR}/lower_level/lib/" )
#include_directories("${PROJECT_SOURCE_DIR}/common/include/")
#include_directories("${PROJECT_SOURCE_DIR}/common/include/comm")
#include_directories("${PROJECT_SOURCE_DIR}/common/include/utilities")

#file(GLOB_RECURSE sources "src/*.cpp")                   # common library cpp files

#add_library(wbc_lib SHARED ${sources})       # produce a library used by others
##aux_source_directory(. sources)
##ADD_LIBRARY ( xp_mpc_lib ${sources})
#target_link_libraries(wbc_lib commonWBC_lib xp_common_lib pthread rt WBIC_lib WBC_Ctrl_lib)
