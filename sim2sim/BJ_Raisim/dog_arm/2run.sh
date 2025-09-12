#!/bin/bash

# 设置库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/task:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/common

# 启动 raisim 可视化环境
gnome-terminal --title="raisim" -- bash -c "cd ~/raisim_workspace/raisimLib/raisimUnity/linux && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib && ./raisimUnity.x86_64; exec bash"

# 等待一下让 raisim 启动
sleep 3

# 启动机器人仿真
{
gnome-terminal -t "dog_sim" -- bash -c "cd /home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/task:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/common && ./dog_sim2; exec bash"
}&

# 等待一下让仿真启动
sleep 2

# 启动执行器
{
gnome-terminal -t "dog_actuator" -- bash -c "cd /home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/task:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/common && ./dog_actuator; exec bash"
}&







