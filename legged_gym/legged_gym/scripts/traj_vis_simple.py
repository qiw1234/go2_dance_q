from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
import os
import torch
import sys
import glob
import json

# Simple motion loader without circular imports
class SimpleMotionLoader:
    def __init__(self, device, motion_files_path):
        self.device = device
        self.trajectory_names = []
        self.trajectory_data = []
        self.trajectory_lens = []
        
        # Load motion files
        motion_files = glob.glob(os.path.join(motion_files_path, "*.json"))
        
        for file_path in motion_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.trajectory_names.append(os.path.basename(file_path))
                self.trajectory_data.append(data)
                self.trajectory_lens.append(len(data['frames']) * data['frame_duration'])
                
        print(f"Loaded {len(self.trajectory_names)} motion files:")
        for name in self.trajectory_names:
            print(f"  - {name}")
    
    def get_frame_at_time(self, motion_id, time):
        if motion_id >= len(self.trajectory_data):
            return None
            
        data = self.trajectory_data[motion_id]
        frame_duration = data['frame_duration']
        frames = data['frames']
        
        frame_idx = int(time / frame_duration) % len(frames)
        return frames[frame_idx]
    
    def get_root_pos(self, frame):
        return torch.tensor([frame[0], frame[1], frame[2]], device=self.device)
    
    def get_root_rot(self, frame):
        return torch.tensor([frame[3], frame[4], frame[5], frame[6]], device=self.device)
    
    def get_joint_pos(self, frame):
        return torch.tensor(frame[25:37], device=self.device)

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="traj visualization")
# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 50.0
sim_params.substeps = 1
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Unable to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# load assets
asset_root = LEGGED_GYM_ROOT_DIR + '/resources/robots'
robot_asset_file = "go2/urdf/go2.urdf"
asset_options = gymapi.AssetOptions()
asset_options.disable_gravity = True
asset_options.collapse_fixed_joints = True
asset_options.flip_visual_attachments = True
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)

robot_body_names = gym.get_asset_rigid_body_names(robot_asset)
robot_dof_names = gym.get_asset_dof_names(robot_asset)
robot_body_nums = len(robot_body_names)
robot_dof_nums = len(robot_dof_names)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# set up the env grid
num_envs = 1
env_lower = gymapi.Vec3(0., 0., 0.)
env_upper = gymapi.Vec3(0., 0., 0.)

# position the camera
cam_pos = gymapi.Vec3(0, -5.0, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# subscribe to keyboard events
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "last mocap")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "next mocap")

# cache useful handles
envs = []
robot_handles = []

for i in range(num_envs):
    # create env instance
    env_handle = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(num_envs)))
    start_pose = gymapi.Transform()
    start_pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
    start_pose.p = gymapi.Vec3(1, -1, 0.3)
    # robot
    robot_handle = gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, 1, 0)
    for body_index in range(robot_body_nums):
        gym.set_rigid_body_color(env_handle, robot_handle, body_index, gymapi.MESH_VISUAL,
                                 gymapi.Vec3(61./255, 132./255, 168./255))

    envs.append(env_handle)
    robot_handles.append(robot_handle)

gym.prepare_sim(sim)

# load the robot data
mocap_motion_files_path = 'opti_traj/output_json'
mocap_motion_loader = SimpleMotionLoader(device=device, motion_files_path=mocap_motion_files_path)

frames = 0
motion_id = 0
actor_count = gym.get_actor_count(envs[0])
actor_ids = torch.arange(actor_count*num_envs, device=device)

print("\n控制说明:")
print("Q键 - 切换到上一个动作")
print("E键 - 切换到下一个动作")
print("ESC键 - 退出")

while not gym.query_viewer_has_closed(viewer):
    # get current frame
    current_time = frames * sim_params.dt
    frame = mocap_motion_loader.get_frame_at_time(motion_id, current_time)
    
    if frame is not None:
        for i in range(num_envs):
            root_vel = torch.tensor([0, 0, 0], device=device)
            root_angular_vel = torch.tensor([0, 0, 0], device=device)

            robot_pos = mocap_motion_loader.get_root_pos(frame)
            robot_quat = mocap_motion_loader.get_root_rot(frame)
            robot_states = torch.cat((robot_pos, robot_quat, root_vel, root_angular_vel), dim=0).to(dtype=torch.float32)

            actor_ids_int32 = actor_ids.to(dtype=torch.int32)
            gym.set_actor_root_state_tensor_indexed(sim,
                                                    gymtorch.unwrap_tensor(robot_states),
                                                    gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
            
            # set robot dof states
            robot_dof_pos = mocap_motion_loader.get_joint_pos(frame)
            robot_dof_vel = torch.zeros_like(robot_dof_pos)
            robot_dof_states = torch.stack((robot_dof_pos, robot_dof_vel), dim=1)

            gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(robot_dof_states), 
                                             gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

    gym.refresh_actor_root_state_tensor(sim)

    frames = (frames + 1) % int(mocap_motion_loader.trajectory_lens[motion_id] / sim_params.dt)
    
    # keyboard event
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "last mocap" and evt.value > 0:
            motion_id = (motion_id - 1) % len(mocap_motion_loader.trajectory_names)
            frames = 0
            print(f"切换到: {mocap_motion_loader.trajectory_names[motion_id]}")
        elif evt.action == "next mocap" and evt.value > 0:
            motion_id = (motion_id + 1) % len(mocap_motion_loader.trajectory_names)
            frames = 0
            print(f"切换到: {mocap_motion_loader.trajectory_names[motion_id]}")
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)