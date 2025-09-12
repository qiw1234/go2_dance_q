#!/bin/bash
# 启动 raisim 可视化环境
gnome-terminal --title="raisim" -- bash -c "cd ~/raisim_workspace/raisimLib/raisimUnity/linux && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib && ./raisimUnity.x86_64; exec bash"

#sleep 5s
#./dog_sim
{
gnome-terminal -t "go2_sim" -- bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/task:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/common:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/sim && export QT_QPA_PLATFORM=offscreen && export QT_LOGGING_RULES=\"*.debug=false;qt.qpa.*=false\" && ./go2_sim;exec bash"
}&


# #./dog_sim_arm_6dof
# {
# gnome-terminal -t "dog_sim_arm_6dof" -- bash -c "./dog_sim_arm_6dof;exec bash"
# }&

sleep 2s
#./dog_actuator
{
gnome-terminal -t "dog_actuator" -- bash -c "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/raisim_build/lib:/home/ubuntu/raisim_workspace/raisimLib/raisim/linux/lib:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/task:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/common:/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/dog_arm/build/sim && ./dog_actuator;exec bash"
}&







