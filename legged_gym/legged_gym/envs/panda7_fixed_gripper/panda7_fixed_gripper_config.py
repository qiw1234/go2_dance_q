from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class panda7_fixed_gripper_BeatCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 2

            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 1.
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 5
            # 机械臂
            track_arm_dof_pos = 1
            track_arm_dof_vel = 1
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_beat.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_BeatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_fixed_gripper_beat')
        # max_iterations = 16000
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'


class panda7_fixed_gripper_TrotCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -10.
            lin_vel_z = -1.0
            feet_air_time = 0

            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            # 机械臂
            track_arm_dof_pos = 0
            track_arm_dof_vel = 5
            track_arm_pos = 5
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_trot.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_TrotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_fixed_gripper_trot')


class panda7_fixed_gripper_SwingCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 0
            test = 0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0.5
            track_root_rot = 2.
            orientation = 0.
            track_lin_vel_ref = 1
            track_ang_vel_ref = 1
            track_dof_pos = 1
            track_dof_vel = 1
            track_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 0
            track_arm_dof_vel = 2
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_swing.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_SwingCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_fixed_gripper_swing')
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_swing/Dec08_17-19-33_/model_10000.pt'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_swing/Dec08_15-06-19_/model_2550.pt'


class panda7_fixed_gripper_Turn_and_jumpCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 0
            test = 0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1
            track_root_height = 0
            track_root_rot = 1
            track_lin_vel_ref = 1
            track_ang_vel_ref = 1
            track_dof_pos = 10
            track_dof_vel = 1
            track_toe_pos = 20
            # jump reward
            jump = 1
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 0
            track_arm_dof_vel = 1
            track_arm_pos = 1
            track_arm_rot = 1

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_turn_and_jump.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_Turn_and_jumpCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_fixed_gripper_turn_and_jump')
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_turn_and_jump/Dec07_00-31-52_/model_7450.pt'


class panda7_fixed_gripper_WaveCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            orientation = -0.0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 1.
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 10
            track_dof_vel = 0
            track_toe_pos = 50
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 0
            track_arm_dof_vel = 1
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_wave.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_WaveCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_fixed_gripper_wave')


class panda7_fixed_gripper_PaceCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 1
            test = 0
            orientation = -0.0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_pace.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_PaceCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7_fixed_gripper_pace')

class panda7_fixed_gripper_SpacetrotCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]

            'arm_joint1': 0,
            'arm_joint2': 0,
            'arm_joint3': 0,
            'arm_joint4': 0,
            'arm_joint5': 0,
            'arm_joint6': 0,
            'arm_joint7': 0,
            'arm_joint8': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 600., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 1., 'joint6': 1}  # [N*m*s/rad] 0.8 1 1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.25,]*12 + [0, 0, 0, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_fixed_gripper/urdf/panda7_nleg_arm_1008_1.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        class scales(LeggedRobotCfg.rewards.scales):
            # regularization reward
            torques = -0.00001
            dof_pos_limits = -10.0
            action_rate = -0.1
            collision = -5.
            lin_vel_z = -1.0
            feet_air_time = 0
            survival = 0
            test = 0
            orientation = -0.0
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0
            track_root_rot = 1
            orientation = 0.
            track_lin_vel_ref = 1
            track_ang_vel_ref = 1
            track_dof_pos = 1
            track_dof_vel = 1
            track_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 3
            track_griper_dof_pos = 0
            track_arm_dof_vel = 3
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper/panda_spacetrot.txt"
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 18
        num_observations = 63
        # debug = True


class panda7_fixed_gripper_SpacetrotCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'panda7_fixed_gripper_Spacetrot'
