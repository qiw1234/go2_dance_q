from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


#----------------------panda with arm--------------------------------
class pandaCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1,  # [rad]

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
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.,
                     'joint7': 10., 'joint8': 10.}  # [N*m/rad]#
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 12., 'joint2': 12., 'joint3': 12.,
                   'joint4': 0.8, 'joint5': 1., 'joint6': 1.,
                   'joint7': 1., 'joint8': 1.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_nleg_arm.urdf'
        name = "panda7"
        foot_name = "FOOT"
        arm_name = "arm_link6"
        penalize_contacts_on = ["thigh", "calf", "base", "arm_link0", "arm_link1", "arm_link2",
                                "arm_link3", "arm_link4", "arm_link5", "arm_link6"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.55
        only_positive_rewards = False
        max_contact_force = 500

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
            delta_torques = -1e-5
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
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = None
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 20
        num_observations = 141
        # debug = True


class pandaCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = ('panda7')
        resume_path = None


class panda7BeatCfg(pandaCfg):
    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_beat.txt"


class panda7BeatCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = ('panda7_beat')
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'


class panda7TrotCfg(pandaCfg):
    class rewards(pandaCfg.rewards):
        class scales(pandaCfg.rewards.scales):
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
            track_griper_dof_pos = 5
            track_arm_dof_vel = 5
            track_arm_pos = 5
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_trot.txt"


class panda7TrotCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = ('panda7_trot')
        # resume_path = 'legged_gym/logs/panda7_beat/Dec01_20-31-14_/model_1500.pt'


class panda7SwingCfg(pandaCfg):
    class rewards(pandaCfg.rewards):
        class scales(pandaCfg.rewards.scales):
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0.5
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_swing.txt"


class panda7SwingCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = ('panda7_swing')


class panda7Turn_and_jumpCfg(pandaCfg):
    class rewards(pandaCfg.rewards):
        class scales(pandaCfg.rewards.scales):
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1
            track_root_height = 0
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1.5
            # jump reward
            jump = 1
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_turn_and_jump.txt"


class panda7Turn_and_jumpCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = ('panda7_turn_and_jump')


class panda7WaveCfg(pandaCfg):
    class rewards(pandaCfg.rewards):
        class scales(pandaCfg.rewards.scales):
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 1.
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 8
            # 机械臂
            track_arm_dof_pos = 1
            track_griper_dof_pos = 5
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(LeggedRobotCfg.env):
        motion_files = "opti_traj/output_panda/panda_wave.txt"


class panda7WaveCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = ('panda7_wave')


class panda7PaceCfg(pandaCfg):
    class rewards(pandaCfg.rewards):
        class scales(pandaCfg.rewards.scales):
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
        motion_files = "opti_traj/output_panda/panda_pace.txt"


class panda7PaceCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = ('panda7_pace')


#----------------------panda fixed gripper--------------------------------
class panda7FixedGripperCfg(pandaCfg):
    class control(pandaCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # 'hip': 128, 'thigh': 144., 'calf': 458,
        # 'hip': 62., 'thigh': 72., 'calf': 229.,

        stiffness = {'hip': 150., 'thigh': 150., 'calf': 150.,
                     'joint1': 150., 'joint2': 150., 'joint3': 150,
                     'joint4': 20., 'joint5': 15., 'joint6': 10.}  # [N*m/rad]# 20  15  10
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0,
                   'joint1': 2., 'joint2': 2, 'joint3': 2,
                   'joint4': 0.1, 'joint5': 0.1, 'joint6': 0.1}  # [N*m*s/rad] 0.8 1 1


        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 50Hz

    class asset(pandaCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_fixed_gripper.urdf'

    class env(pandaCfg.env):
        num_actions = 18
        num_observations = 60  #60
        num_leg = 4
        motion_files = "opti_traj/output_panda_fixed_gripper_json"


class panda7FixedGripperBeatCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
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

    class env(panda7FixedGripperCfg.env):
        # motion_files = "opti_traj/output_panda_fixed_gripper/panda_beat.txt"
        motion_name = 'beat'


class panda7FixedGripperBeatCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_beat'


class panda7FixedGripperTrotCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1
            track_root_height = 0
            track_root_rot = 1
            track_lin_vel_ref = 1
            track_ang_vel_ref = 1
            track_dof_pos = 5
            track_dof_vel = 1
            track_toe_pos = 5
            # 机械臂
            track_arm_dof_pos = 3
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        # motion_files = "opti_traj/output_panda_fixed_gripper/panda_trot.txt"
        motion_name = 'panda_trot'


class panda7FixedGripperTrotCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_trot'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_trot/Dec26_22-45-05_/model_23000.pt'

class panda7FixedGripperPaceCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 3
            track_root_height = 0
            track_root_rot = 1
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 10
            track_dof_vel = 1
            track_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 10
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        # motion_files = "opti_traj/output_panda_fixed_gripper/panda_pace.txt"
        motion_name = 'panda_pace'


class panda7FixedGripperPaceCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_pace'

class panda7FixedGripperSwingCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 0
            track_root_height = 0.5
            track_root_rot = 0.
            orientation = -0.1
            tracking_yaw = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 5
            track_dof_vel = 0
            track_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 10
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        motion_name = 'swing'


class panda7FixedGripperSwingCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_swing'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_swing/Dec28_12-06-32_/model_34500.pt'

class panda7FixedGripperWaveCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            rf_no_action = -1
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1.
            track_root_height = 0.
            track_root_rot = 1.2
            orientation = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 1.5
            track_dof_vel = 0
            track_toe_pos = 10
            # track_LF_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 5
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        # motion_files = "opti_traj/output_panda_fixed_gripper/panda_wave.txt"
        motion_name = 'wave'



class panda7FixedGripperWaveCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_wave'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_wave/Dec27_11-31-46_/model_103000.pt'

class panda7FixedGripperTurnAndJumpCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            lin_vel_z = -0
            survival = 1
            feet_air_time = 1
            feet_contact_time = 0
            feet_contact_forces = -0.05
            action_rate = -0.3
            # arm_dof_error = -2.
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1
            track_root_height = 0
            track_root_rot = 10
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 5
            track_dof_vel = 0
            track_toe_pos = 0
            track_toe_height = 5
            track_toe_x = 0
            track_toe_y = 0
            # jump reward
            jump = 20.
            # 机械臂
            track_arm_dof_pos = 10 #20
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        # motion_files = "opti_traj/output_panda_fixed_gripper/panda_turn_and_jump.txt"
        motion_name = 'turn_and_jump'

    class domain_rand(panda7FixedGripperCfg.domain_rand):
        RSI = False



class panda7FixedGripperTurnAndJumpCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_turn_and_jump'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_turn_and_jump/Jan08_15-07-28_/model_35000.pt'

class panda7FixedGripperSpaceTrotCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            feet_air_time = 3.
            leg_num_contact = -2.
            feet_contact_time = 3.
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 3
            track_root_height = 0
            track_root_rot = 1
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 10
            track_dof_vel = 0
            track_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 8
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        # motion_files = "opti_traj/output_panda_fixed_gripper/panda_spacetrot.txt"
        motion_name = 'spacetrot'


class panda7FixedGripperSpaceTrotCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_spacetrot'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_spacetrot/Dec15_17-59-23_/model_8000.pt'
        # resume_path = 'legged_gym/logs/panda7_fixed_gripper_spacetrot/Dec15_10-37-34_/model_29400.pt' # 滑步

class panda7FixedGripperStandCfg(panda7FixedGripperCfg):

    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            feet_air_time = 0
            leg_num_contact = 0
            feet_contact_time = 0
            feet_contact_forces = -0.05
            action_rate = -0.3
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 3
            track_root_height = 0
            track_root_rot = 10
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 8
            track_dof_vel = 0
            track_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 8
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class env(panda7FixedGripperCfg.env):
        motion_name = 'stand'
        motion_files = "opti_traj/STANDTraj"
        check_contact = False


class panda7FixedGripperStandCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_stand'

class panda7FixedGripperArmLegCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            rf_no_action = -1
            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 1.
            track_root_height = 0.
            track_root_rot = 1.2
            orientation = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 1.5
            track_dof_vel = 0
            track_toe_pos = 10
            # track_LF_toe_pos = 10
            # 机械臂
            track_arm_dof_pos = 10
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0

    class asset(pandaCfg.asset):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7_nleg_rm_arm/panda7_nleg_rm_arm_0102.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/panda7/urdf/panda7_fixed_gripper.urdf'

    class env(panda7FixedGripperCfg.env):
        motion_name = 'leg_with_arm'



class panda7FixedGripperArmLegCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_arm_leg'


#----------------------panda7 fixed gripper trans config---------------------------
class panda7FixedGripperTransCfg(panda7FixedGripperCfg):
    class rewards(panda7FixedGripperCfg.rewards):
        class scales(panda7FixedGripperCfg.rewards.scales):
            # 机械臂
            track_arm_dof_pos = 0
            track_griper_dof_pos = 0
            track_arm_dof_vel = 0
            track_arm_pos = 0
            track_arm_rot = 0
    class env(panda7FixedGripperCfg.env):
        motion_files = "opti_traj/output_panda_fixed_gripper_json"
        dance_sequence = None
        # RSI = False
        motion_name = 'spacetrot' # 这里随便写一个，用于选择轨迹，实际上没用
        episode_length_s = 20

class panda7FixedGripperTransCfgPPO(pandaCfgPPO):
    class runner(pandaCfgPPO.runner):
        experiment_name = 'panda7_fixed_gripper_trans'
        resume_path = 'legged_gym/logs/panda7_fixed_gripper_turn_and_jump/Dec11_20-30-49_/model_750.pt' # 随便给个路径就行，这个路径的文件不用
        dance_task_list = ['panda7_fixed_gripper_wave', 'panda7_fixed_gripper_spacetrot', 'panda7_fixed_gripper_trot',
                           'panda7_fixed_gripper_turn_and_jump',
                           'panda7_fixed_gripper_swing', 'panda7_fixed_gripper_beat', 'panda7_fixed_gripper_pace',]
