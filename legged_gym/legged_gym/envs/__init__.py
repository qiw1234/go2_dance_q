# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


from .base.legged_robot import LeggedRobot
from .go2.go2_dance_config import GO2DanceCfg_swing, GO2DanceCfg_swingPPO

from .go2.go2_dance_config import GO2DanceCfg_beat, GO2DanceCfg_beatPPO
from .go2.go2_dance_config import GO2DanceCfg_turn_and_jump, GO2DanceCfg_turn_and_jumpPPO
from .go2.go2_dance_config import GO2DanceCfg_wave, GO2DanceCfg_wavePPO
from .go2.go2_dance_config import GO2DanceCfg_pace, GO2DanceCfg_pacePPO
from .go2.go2_dance_config import GO2DanceCfg_trot, GO2DanceCfg_trotPPO
from .go2.go2_dance_config import GO2DanceCfg_stand, GO2DanceCfg_standPPO
from .go2.go2_dance_config import GO2DanceCfg_sidestep, GO2DanceCfg_sidestepPPO


from legged_gym.utils.task_registry import task_registry

# GO2
task_registry.register( "go2_swing", LeggedRobot, GO2DanceCfg_swing(), GO2DanceCfg_swingPPO() )
task_registry.register( "go2_beat", LeggedRobot, GO2DanceCfg_beat(), GO2DanceCfg_beatPPO() )
task_registry.register( "go2_turn_and_jump", LeggedRobot, GO2DanceCfg_turn_and_jump(),
                        GO2DanceCfg_turn_and_jumpPPO() )
task_registry.register( "go2_wave", LeggedRobot, GO2DanceCfg_wave(), GO2DanceCfg_wavePPO() )
task_registry.register( "go2_pace", LeggedRobot, GO2DanceCfg_pace(), GO2DanceCfg_pacePPO() )
task_registry.register( "go2_trot", LeggedRobot, GO2DanceCfg_trot(), GO2DanceCfg_trotPPO() )
task_registry.register("go2_stand", LeggedRobot, GO2DanceCfg_stand(), GO2DanceCfg_standPPO())
task_registry.register("go2_sidestep", LeggedRobot, GO2DanceCfg_sidestep(), GO2DanceCfg_sidestepPPO())
