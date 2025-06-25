#!/bin/bash

#./dog_sim_raisim
{
gnome-terminal -t "dog_sim_unity" -- bash -c "~/raisim/raisimLib/raisimUnityOpengl/linux/raisimUnity.x86_64;exec bash"
}&
#./dog_ocu
#cd ../upper
#{
#gnome-terminal -t "dog_ocu" -- bash -c "./dog_ocu;exec bash"
#}&
