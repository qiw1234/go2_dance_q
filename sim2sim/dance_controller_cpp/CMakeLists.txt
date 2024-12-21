cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(DOGDANCE)

set(Torch_DIR /usr/local/lib/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# include_directories("/usr/include/eigen3" $"{TORCH_INCLUDE_DIRS}")

add_executable(bjtu_dance bjtu_dance.cpp)
target_link_libraries(bjtu_dance "${TORCH_LIBRARIES}")
set_property(TARGET bjtu_dance PROPERTY CXX_STANDARD 17)