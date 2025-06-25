#!/bin/bash
gnome-terminal --title="raisim" -- bash -c "cd ~/raisim/raisim_workspace/raisimLib-1.1.7/raisimUnity/linux;./raisimUnity.x86_64"
#sleep 5s
#./dog_sim
{
gnome-terminal -t "panda7_sim" -- bash -c "./panda7_sim;exec bash"
}&

# #./dog_sim_arm_6dof
# {
# gnome-terminal -t "dog_sim_arm_6dof" -- bash -c "./dog_sim_arm_6dof;exec bash"
# }&

sleep 2s
#./dog_actuator
{
gnome-terminal -t "dog_actuator" -- bash -c "./dog_actuator;exec bash"
}&







