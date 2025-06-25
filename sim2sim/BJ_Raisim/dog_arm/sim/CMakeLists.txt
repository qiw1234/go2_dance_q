if(USING_RAISIM_ENABLE)


# Source/Include files
    include_directories(${CMAKE_BINARY_DIR})
    include_directories(${PROJECT_SOURCE_DIR})

    include_directories("/usr/include/eigen3/") # common includes
    include_directories("${PROJECT_SOURCE_DIR}/common/include/")
    include_directories("${PROJECT_SOURCE_DIR}/api_sdk/include/")
    file(GLOB_RECURSE libsources "${PROJECT_SOURCE_DIR}/api_sdk/lib/*.so")
    include_directories("${PROJECT_SOURCE_DIR}/lower_level/include/")
    include_directories("${PROJECT_SOURCE_DIR}/mid_level/include/")
    include_directories("${PROJECT_SOURCE_DIR}/upper_level/include/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/qpOASES/include/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/EphCan/include")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/dynacore_yaml-cpp/")
    include_directories("${PROJECT_SOURCE_DIR}/sim/include/Raisim/")

    file(GLOB_RECURSE sources "src/xp_sim.cpp")                   # common library cpp files
    #_USE_ARM
    include_directories("${PROJECT_SOURCE_DIR}/third_party/rmARM/include")
    file(GLOB_RECURSE libsources_arm "${PROJECT_SOURCE_DIR}/third_party/rmARM/lib/*.so")

    aux_source_directory(. sources)
    ADD_LIBRARY ( xp_sim_lib SHARED ${sources})

    target_link_libraries(xp_sim_lib xp_common_lib pthread rt qpOASES inih)
    target_link_libraries(xp_sim_lib ${libsources})
    target_link_libraries(xp_sim_lib ${libsources_arm})

else(USING_RAISIM_ENABLE)

    SET(CMAKE_INCLUDE_CURRENT_DIR ON)
    execute_process(COMMAND ${CMAKE_SOURCE_DIR}/scripts/find_qt_path.sh OUTPUT_VARIABLE CMAKE_PREFIX_PATH_1)
    SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH_1})
    # message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

    if(POLICY CMP0071)
      cmake_policy(SET CMP0071 OLD)
    endif()

    # Instruct CMake to run moc automatically when needed
    set(CMAKE_AUTOMOC ON)
    # Create code from a list of Qt designer ui files
    set(CMAKE_AUTOUIC ON)
    set(Qt5Core_DIR /opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake/Qt5Core)
    set(Qt5Widgets_DIR /opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake/Qt5Widgets)
    set(Qt5Gamepad_DIR /opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake/Qt5Gamepad)
    find_package(Qt5Core CONFIG REQUIRED)
    find_package(Qt5Widgets CONFIG REQUIRED)
    get_target_property(QtCore_location Qt5::Core LOCATION)
    message(STATUS ${QtCore_location})
    find_package(Qt5Gui CONFIG REQUIRED)

    set(QOBJECT_HEADERS
            include/Graphics3D.h
            include/SimControlPanel.h)

    QT5_WRAP_CPP(tst_hdr_moc ${QOBJECT_HEADERS})

    include_directories(".")
    include_directories("include/" )
    include_directories("${PROJECT_SOURCE_DIR}/common/include/")
    include_directories("${PROJECT_SOURCE_DIR}/lower_level/include/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/qpOASES/include/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/inih/")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/ParamYaml/dynacore_yaml-cpp/")
    include_directories("/usr/local/include/lcm/")   # lcm includes
    include_directories(${CMAKE_BINARY_DIR})
    include_directories("${PROJECT_SOURCE_DIR}/wbc/CommonWbc/include")
    include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/ControlParameters")
    include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/Utilities")
    include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/SimUtilities")
    include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim/Collision")
    include_directories("${PROJECT_SOURCE_DIR}/sim/include/comm_sim")
    file(GLOB sources "src/*.cpp")
    if(USING_RAISIM_ENABLE)
    include_directories("${TORCH_INCLUDE_DIRS}")
    endif(USING_RAISIM_ENABLE)

    find_package(Qt5Core COMPONENTS QtGamepad REQUIRED)
    find_package(Qt5Gamepad REQUIRED)

    set(OpenGL_GL_PREFERENCE LEGACY)

    find_package(OpenGL REQUIRED)
    include_directories(${OPENGL_INCLUDE_DIR})



    #add_executable(sim ${sources} ${tst_hdr_moc})

    aux_source_directory(. sources)
    ADD_LIBRARY ( xp_sim_lib SHARED ${sources} ${tst_hdr_moc})

    target_link_libraries(xp_sim_lib pthread lcm
        Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Gamepad ${QT_LIBRARIES} ${OPENGL_LIBRARIES})
    target_link_libraries(xp_sim_lib commonWBC_lib inih)



    if (CMAKE_SYSTEM_NAME MATCHES Linux)
      target_link_libraries(xp_sim_lib rt ${GLUT_glut_LIBRARIES})
      include (CheckIncludeFileCXX)
      ADD_DEFINITIONS(-Dlinux)
    endif (CMAKE_SYSTEM_NAME MATCHES Linux)

    if (APPLE)
      target_link_libraries(xp_sim_lib "-framework GLUT")
    endif (APPLE)

    target_link_libraries(xp_sim_lib ${TORCH_LIBRARIES})



endif(USING_RAISIM_ENABLE)
