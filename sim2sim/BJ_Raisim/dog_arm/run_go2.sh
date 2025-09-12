#!/bin/bash
# 启动 raisim 可视化环境
#!/usr/bin/env bash
set -e

# 1) kill old sim
pkill -f go2_sim 2>/dev/null || true
pkill -f RaisimServer 2>/dev/null || true
pkill -f DogSimRaisim2 2>/dev/null || true

# 2) clean SysV IPC (owned by当前用户)
for id in $(ipcs -m | awk 'NR>3 {print $2}'); do ipcrm -m "$id" 2>/dev/null || true; done
for id in $(ipcs -s | awk 'NR>3 {print $2}'); do ipcrm -s "$id" 2>/dev/null || true; done

# 3) optional: logs
rm -f sim2sim/BJ_Raisim/net/HSW/data/*.csv 2>/dev/null || true

# 4) run your sim (原来的启动命令)！！！！！注意要先运行环境再运行推理脚本
# ./go2_sim 或 ./build/xxx

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







