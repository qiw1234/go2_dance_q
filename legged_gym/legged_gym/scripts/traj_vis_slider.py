#!/usr/bin/env python3
"""
轨迹可视化脚本 - 带滑条控制
基于traj_vis_simple.py重新设计，支持帧数显示和滑条控制
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
import os
import torch
import glob
import json
import tkinter as tk
from tkinter import ttk
import threading
import time

class MotionLoader:
    """运动数据加载器"""
    def __init__(self, device, motion_files_path):
        self.device = device
        self.motions = []
        
        # 加载所有JSON文件
        json_files = glob.glob(os.path.join(motion_files_path, "*.json"))
        json_files.sort()  # 排序确保顺序一致
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    motion_info = {
                        'name': os.path.basename(file_path),
                        'frames': data['frames'],
                        'frame_duration': data['frame_duration'],
                        'total_frames': len(data['frames']),
                        'total_time': len(data['frames']) * data['frame_duration']
                    }
                    self.motions.append(motion_info)
                    print(f"✓ 加载: {motion_info['name']} ({motion_info['total_frames']} 帧)")
            except Exception as e:
                print(f"✗ 加载失败 {file_path}: {e}")
        
        if not self.motions:
            raise ValueError("未找到有效的轨迹文件！")
        
        print(f"总共加载了 {len(self.motions)} 个轨迹文件")
    
    def get_motion_names(self):
        return [motion['name'] for motion in self.motions]
    
    def get_frame_data(self, motion_id, frame_id):
        """获取指定动作的指定帧数据"""
        if 0 <= motion_id < len(self.motions) and 0 <= frame_id < self.motions[motion_id]['total_frames']:
            return self.motions[motion_id]['frames'][frame_id]
        return None
    
    def get_motion_info(self, motion_id):
        """获取动作信息"""
        if 0 <= motion_id < len(self.motions):
            return self.motions[motion_id]
        return None

class TrajectoryController:
    """轨迹控制器GUI"""
    def __init__(self, motion_loader):
        self.motion_loader = motion_loader
        self.current_motion = 0
        self.current_frame = 0
        self.is_playing = False
        self.play_speed = 1.0
        self.gui_active = False
        
        # 创建GUI
        try:
            self.setup_gui()
            self.gui_active = True
            print("GUI控制面板已启动")
        except Exception as e:
            print(f"GUI启动失败: {e}")
            print("将使用键盘控制模式")
    
    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("轨迹播放控制器")
        self.root.geometry("650x200")
        self.root.attributes('-topmost', True)
        
        # 动作选择
        frame1 = tk.Frame(self.root)
        frame1.pack(pady=10, padx=20, fill='x')
        
        tk.Label(frame1, text="动作:", font=('Arial', 10, 'bold')).pack(side='left')
        self.motion_var = tk.StringVar()
        self.motion_combo = ttk.Combobox(frame1, textvariable=self.motion_var,
                                        values=self.motion_loader.get_motion_names(),
                                        state='readonly', width=25)
        self.motion_combo.pack(side='left', padx=10)
        self.motion_combo.current(0)
        self.motion_combo.bind('<<ComboboxSelected>>', self.on_motion_change)
        
        # 播放控制
        frame2 = tk.Frame(self.root)
        frame2.pack(pady=5, padx=20, fill='x')
        
        self.play_btn = tk.Button(frame2, text="播放", command=self.toggle_play, width=8)
        self.play_btn.pack(side='left', padx=5)
        
        tk.Button(frame2, text="重置", command=self.reset, width=8).pack(side='left', padx=5)
        tk.Button(frame2, text="◀", command=self.prev_frame, width=3).pack(side='left', padx=2)
        tk.Button(frame2, text="▶", command=self.next_frame, width=3).pack(side='left', padx=2)
        
        # 速度控制
        tk.Label(frame2, text="速度:").pack(side='left', padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(frame2, from_=0.1, to=3.0, resolution=0.1,
                              variable=self.speed_var, orient='horizontal', length=100)
        speed_scale.pack(side='left', padx=5)
        speed_scale.bind('<Motion>', self.on_speed_change)
        
        # 帧滑条
        frame3 = tk.Frame(self.root)
        frame3.pack(pady=10, padx=20, fill='x')
        
        tk.Label(frame3, text="帧控制:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.frame_var = tk.IntVar()
        motion_info = self.motion_loader.get_motion_info(0)
        max_frame = motion_info['total_frames'] - 1 if motion_info else 0
        
        self.frame_scale = tk.Scale(frame3, from_=0, to=max_frame,
                                   variable=self.frame_var, orient='horizontal', length=600)
        self.frame_scale.pack(fill='x', pady=5)
        self.frame_scale.bind('<Motion>', self.on_frame_change)
        self.frame_scale.bind('<ButtonRelease-1>', self.on_frame_change)
        
        # 信息显示
        frame4 = tk.Frame(self.root)
        frame4.pack(pady=5, padx=20, fill='x')
        
        self.info_label = tk.Label(frame4, text="", bg='lightgray', relief='sunken',
                                  font=('Arial', 10), anchor='w', padx=10, pady=5)
        self.info_label.pack(fill='x')
        
        self.update_info()
    
    def update_gui(self):
        """更新GUI（非阻塞）"""
        if self.gui_active:
            try:
                self.root.update()
            except:
                self.gui_active = False
    
    def on_motion_change(self, event=None):
        """动作切换"""
        new_motion = self.motion_combo.current()
        if new_motion != self.current_motion:
            self.current_motion = new_motion
            self.current_frame = 0
            self.frame_var.set(0)
            
            # 更新滑条范围
            motion_info = self.motion_loader.get_motion_info(self.current_motion)
            if motion_info:
                self.frame_scale.config(to=motion_info['total_frames'] - 1)
            
            self.update_info()
            print(f"切换到: {self.motion_loader.get_motion_names()[self.current_motion]}")
    
    def on_frame_change(self, event=None):
        """帧滑条变化"""
        new_frame = self.frame_var.get()
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.update_info()
    
    def on_speed_change(self, event=None):
        """速度变化"""
        self.play_speed = self.speed_var.get()
    
    def toggle_play(self):
        """播放/暂停切换"""
        self.is_playing = not self.is_playing
        self.play_btn.config(text="暂停" if self.is_playing else "播放")
    
    def reset(self):
        """重置到第一帧"""
        self.current_frame = 0
        self.frame_var.set(0)
        self.is_playing = False
        self.play_btn.config(text="播放")
        self.update_info()
    
    def prev_frame(self):
        """上一帧"""
        motion_info = self.motion_loader.get_motion_info(self.current_motion)
        if motion_info:
            self.current_frame = (self.current_frame - 1) % motion_info['total_frames']
            self.frame_var.set(self.current_frame)
            self.update_info()
    
    def next_frame(self):
        """下一帧"""
        motion_info = self.motion_loader.get_motion_info(self.current_motion)
        if motion_info:
            self.current_frame = (self.current_frame + 1) % motion_info['total_frames']
            self.frame_var.set(self.current_frame)
            self.update_info()
    
    def update_info(self):
        """更新信息显示"""
        motion_info = self.motion_loader.get_motion_info(self.current_motion)
        if motion_info:
            current_time = self.current_frame * motion_info['frame_duration']
            info_text = f"帧: {self.current_frame + 1:04d}/{motion_info['total_frames']:04d} | " \
                       f"时间: {current_time:.3f}s/{motion_info['total_time']:.3f}s | " \
                       f"速度: {self.play_speed:.1f}x"
            
            if self.gui_active:
                try:
                    self.info_label.config(text=info_text)
                except:
                    self.gui_active = False
            
            # 在终端也显示信息
            print(f"\r{info_text}", end='', flush=True)
    
    def update_auto_play(self):
        """自动播放更新"""
        if self.is_playing:
            motion_info = self.motion_loader.get_motion_info(self.current_motion)
            if motion_info:
                # 计算下一帧
                step = max(1, int(self.play_speed))
                self.current_frame = (self.current_frame + step) % motion_info['total_frames']
                if self.gui_active:
                    self.frame_var.set(self.current_frame)
                self.update_info()
    
    def get_current_frame_data(self):
        """获取当前帧数据"""
        return self.motion_loader.get_frame_data(self.current_motion, self.current_frame)
    
    def switch_motion(self, direction):
        """切换动作（键盘控制）"""
        total_motions = len(self.motion_loader.motions)
        if direction > 0:
            self.current_motion = (self.current_motion + 1) % total_motions
        else:
            self.current_motion = (self.current_motion - 1) % total_motions
        
        self.current_frame = 0
        if self.gui_active:
            self.motion_combo.current(self.current_motion)
            self.frame_var.set(0)
            motion_info = self.motion_loader.get_motion_info(self.current_motion)
            if motion_info:
                self.frame_scale.config(to=motion_info['total_frames'] - 1)
        
        self.update_info()
        print(f"切换到: {self.motion_loader.get_motion_names()[self.current_motion]}")
    
    def keyboard_control_frame(self, direction):
        """键盘控制帧（左右箭头键）"""
        motion_info = self.motion_loader.get_motion_info(self.current_motion)
        if motion_info:
            if direction > 0:
                self.current_frame = (self.current_frame + 1) % motion_info['total_frames']
            else:
                self.current_frame = (self.current_frame - 1) % motion_info['total_frames']
            
            if self.gui_active:
                self.frame_var.set(self.current_frame)
            self.update_info()

def main():
    print("启动轨迹可视化程序...")
    
    # 初始化Isaac Gym
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="轨迹可视化 - 滑条控制")
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    
    # 配置仿真参数
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
    
    # 创建仿真环境
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise Exception("创建仿真环境失败")
    
    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # 加载机器人资源
    asset_root = LEGGED_GYM_ROOT_DIR + '/resources/robots'
    robot_asset_file = "go2/urdf/go2.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = True
    asset_options.flip_visual_attachments = True
    robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)
    
    # 创建查看器
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("创建查看器失败")
    
    # 设置相机位置
    cam_pos = gymapi.Vec3(0, -5.0, 2)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 注册键盘事件
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "prev_motion")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "next_motion")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_play")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "prev_frame")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next_frame")
    
    # 创建环境和机器人
    num_envs = 1
    envs = []
    robots = []
    
    for i in range(num_envs):
        env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), int(np.sqrt(num_envs)))
        
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)
        pose.p = gymapi.Vec3(0, 0, 0.3)
        
        robot = gym.create_actor(env, robot_asset, pose, "robot", i, 1, 0)
        
        # 设置机器人颜色
        for j in range(gym.get_actor_rigid_body_count(env, robot)):
            gym.set_rigid_body_color(env, robot, j, gymapi.MESH_VISUAL,
                                   gymapi.Vec3(0.24, 0.52, 0.66))
        
        envs.append(env)
        robots.append(robot)
    
    gym.prepare_sim(sim)
    
    # 加载轨迹数据
    motion_path = '/home/ubuntu/robot_dance/opti_traj/output_json'
    print(f"从路径加载轨迹: {motion_path}")
    motion_loader = MotionLoader(device, motion_path)
    
    # 创建控制器
    controller = TrajectoryController(motion_loader)
    
    # 获取actor信息
    actor_count = gym.get_actor_count(envs[0])
    actor_ids = torch.arange(actor_count * num_envs, device=device, dtype=torch.int32)
    
    print("\n=== 控制说明 ===")
    print("GUI面板: 选择动作、播放控制、帧滑条")
    print("键盘: Q/E切换动作, 空格播放/暂停, R重置")
    print("==================\n")
    
    # 主循环
    last_update = time.time()
    update_rate = 50.0  # Hz
    
    try:
        while not gym.query_viewer_has_closed(viewer):
            current_time = time.time()
            
            # 控制更新频率
            if current_time - last_update >= 1.0 / update_rate:
                controller.update_auto_play()
                controller.update_gui()  # 更新GUI
                last_update = current_time
            
            # 获取当前帧数据
            frame_data = controller.get_current_frame_data()
            
            if frame_data is not None:
                # 设置机器人状态
                root_pos = torch.tensor(frame_data[0:3], device=device, dtype=torch.float32)
                root_quat = torch.tensor(frame_data[3:7], device=device, dtype=torch.float32)
                root_vel = torch.zeros(3, device=device, dtype=torch.float32)
                root_ang_vel = torch.zeros(3, device=device, dtype=torch.float32)
                
                root_states = torch.cat([root_pos, root_quat, root_vel, root_ang_vel])
                
                gym.set_actor_root_state_tensor_indexed(sim,
                                                      gymtorch.unwrap_tensor(root_states),
                                                      gymtorch.unwrap_tensor(actor_ids),
                                                      len(actor_ids))
                
                # 设置关节状态
                joint_pos = torch.tensor(frame_data[25:37], device=device, dtype=torch.float32)
                joint_vel = torch.zeros_like(joint_pos)
                joint_states = torch.stack([joint_pos, joint_vel], dim=1)
                
                gym.set_dof_state_tensor_indexed(sim,
                                               gymtorch.unwrap_tensor(joint_states),
                                               gymtorch.unwrap_tensor(actor_ids),
                                               len(actor_ids))
            
            # 处理键盘事件
            for event in gym.query_viewer_action_events(viewer):
                if event.action == "prev_motion" and event.value > 0:
                    controller.switch_motion(-1)
                elif event.action == "next_motion" and event.value > 0:
                    controller.switch_motion(1)
                elif event.action == "toggle_play" and event.value > 0:
                    controller.toggle_play()
                elif event.action == "reset" and event.value > 0:
                    controller.reset()
                elif event.action == "prev_frame" and event.value > 0:
                    controller.keyboard_control_frame(-1)
                elif event.action == "next_frame" and event.value > 0:
                    controller.keyboard_control_frame(1)
            
            # 更新仿真
            gym.refresh_actor_root_state_tensor(sim)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
    
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        print("清理资源...")
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        print("程序结束")

if __name__ == "__main__":
    main()