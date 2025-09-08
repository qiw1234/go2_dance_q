import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import utils

class TrajectoryVisualizer:
    def __init__(self, json_file):
        """
        轨迹可视化器
        Args:
            json_file: JSON轨迹文件路径
        """
        self.load_trajectory(json_file)
        self.go2 = utils.QuadrupedRobot()
        self.setup_plot()
        
    def load_trajectory(self, json_file):
        """加载轨迹数据"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.frame_duration = data['frame_duration']
        self.frames = np.array(data['frames'])
        self.num_frames = len(self.frames)
        self.fps = 1.0 / self.frame_duration
        
        print(f"加载轨迹文件: {json_file}")
        print(f"总帧数: {self.num_frames}")
        print(f"持续时间: {self.num_frames * self.frame_duration:.2f}秒")
        print(f"帧率: {self.fps:.1f}fps")
        
    def setup_plot(self):
        """设置绘图窗口"""
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建主窗口
        self.fig = plt.figure(figsize=(16, 10))
        
        # 3D视图 - 机器人姿态
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_title('Robot Pose (3D View)')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # 俯视图 - 足端轨迹
        self.ax_top = self.fig.add_subplot(222)
        self.ax_top.set_title('Foot Trajectories (Top View)')
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Y (m)')
        self.ax_top.grid(True)
        self.ax_top.set_aspect('equal')
        
        # 侧视图 - 足端高度
        self.ax_side = self.fig.add_subplot(223)
        self.ax_side.set_title('Foot Heights (Side View)')
        self.ax_side.set_xlabel('Y (m)')
        self.ax_side.set_ylabel('Z (m)')
        self.ax_side.grid(True)
        
        # 质心轨迹
        self.ax_com = self.fig.add_subplot(224)
        self.ax_com.set_title('Center of Mass Trajectory')
        self.ax_com.set_xlabel('Time (s)')
        self.ax_com.set_ylabel('Position (m)')
        self.ax_com.grid(True)
        
        # 滑条控制区域
        slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])
        self.slider = Slider(slider_ax, 'Frame', 0, self.num_frames-1, 
                           valinit=0, valfmt='%d', valstep=1)
        self.slider.on_changed(self.update_frame)
        
        # 初始化绘图元素
        self.init_plot_elements()
        
        # 显示第一帧
        self.update_frame(0)
        
    def init_plot_elements(self):
        """初始化绘图元素"""
        # 3D视图中的机器人身体和腿部
        self.body_lines_3d = []
        self.leg_lines_3d = []
        self.foot_points_3d = []
        
        # 足端轨迹线
        self.foot_traj_lines = []
        self.foot_current_points = []
        
        # 足端高度线
        self.foot_height_lines = []
        
        # 质心轨迹线
        self.com_lines = []
        
        # 颜色设置
        self.colors = ['red', 'green', 'blue', 'orange']
        self.leg_names = ['FR', 'FL', 'HR', 'HL']
        
    def get_frame_data(self, frame_idx):
        """获取指定帧的数据"""
        if frame_idx >= self.num_frames:
            frame_idx = self.num_frames - 1
            
        frame = self.frames[frame_idx]
        
        data = {
            'root_pos': frame[:3],
            'root_rot': frame[3:7],
            'root_lin_vel': frame[7:10],
            'root_ang_vel': frame[10:13],
            'toe_pos': frame[13:25].reshape(4, 3),
            'dof_pos': frame[25:37].reshape(4, 3),
            'dof_vel': frame[37:49].reshape(4, 3)
        }
        
        return data
        
    def calculate_foot_positions(self, data):
        """计算足端在世界坐标系下的位置"""
        root_pos = data['root_pos']
        root_rot = data['root_rot']  # quaternion [x, y, z, w]
        dof_pos = data['dof_pos']
        
        # 将四元数转换为旋转矩阵
        if np.linalg.norm(root_rot) > 1e-6:
            rotm = utils.quaternion2rotm(root_rot)
        else:
            rotm = np.eye(3)
        
        foot_positions = []
        for i in range(4):
            # 直接使用正运动学公式计算足端位置，避免使用CasADi符号变量
            foot_local = self.forward_kinematics(dof_pos[i], i)
            
            # 转换到世界坐标系
            foot_world = rotm @ foot_local + root_pos
            foot_positions.append(foot_world)
            
        return np.array(foot_positions)
    
    def forward_kinematics(self, joint_angles, leg_id):
        """
        直接计算正运动学，避免使用CasADi符号变量
        Args:
            joint_angles: [hip_x, hip_y, knee] 关节角度
            leg_id: 腿部ID (0:FR, 1:FL, 2:HR, 3:HL)
        Returns:
            足端在机身坐标系下的位置
        """
        l1, l2, l3 = self.go2.l1, self.go2.l2, self.go2.l3
        L, W = self.go2.L, self.go2.W
        
        q1, q2, q3 = joint_angles
        
        # 计算足端相对于髋关节的位置
        if leg_id == 0 or leg_id == 2:  # 右侧腿
            # 右侧腿的运动学
            x = l2 * np.cos(q2) + l3 * np.cos(q2 + q3)
            y = -l1 * np.cos(q1) - (l2 * np.sin(q2) + l3 * np.sin(q2 + q3)) * np.sin(q1)
            z = -l1 * np.sin(q1) + (l2 * np.sin(q2) + l3 * np.sin(q2 + q3)) * np.cos(q1)
        else:  # 左侧腿
            # 左侧腿的运动学
            x = l2 * np.cos(q2) + l3 * np.cos(q2 + q3)
            y = l1 * np.cos(q1) + (l2 * np.sin(q2) + l3 * np.sin(q2 + q3)) * np.sin(q1)
            z = -l1 * np.sin(q1) + (l2 * np.sin(q2) + l3 * np.sin(q2 + q3)) * np.cos(q1)
        
        # 髋关节相对于机身中心的位置
        if leg_id == 0:  # FR
            hip_pos = np.array([L/2, -W/2, 0])
        elif leg_id == 1:  # FL
            hip_pos = np.array([L/2, W/2, 0])
        elif leg_id == 2:  # HR
            hip_pos = np.array([-L/2, -W/2, 0])
        else:  # HL
            hip_pos = np.array([-L/2, W/2, 0])
        
        # 足端在机身坐标系下的位置
        foot_pos = hip_pos + np.array([x, y, z])
        
        return foot_pos
        
    def draw_robot_body(self, root_pos, root_rot):
        """绘制机器人身体"""
        # 清除之前的身体线条
        for line in self.body_lines_3d:
            line.remove()
        self.body_lines_3d.clear()
        
        # 机身尺寸
        L, W = self.go2.L, self.go2.W
        
        # 机身四个角点（相对于质心）
        corners = np.array([
            [L/2, -W/2, 0],
            [L/2, W/2, 0],
            [-L/2, W/2, 0],
            [-L/2, -W/2, 0]
        ])
        
        # 应用旋转和平移
        if np.linalg.norm(root_rot) > 1e-6:
            rotm = utils.quaternion2rotm(root_rot)
            corners = (rotm @ corners.T).T
            
        corners += root_pos
        
        # 绘制机身轮廓
        body_x = np.append(corners[:, 0], corners[0, 0])
        body_y = np.append(corners[:, 1], corners[0, 1])
        body_z = np.append(corners[:, 2], corners[0, 2])
        
        line = self.ax_3d.plot(body_x, body_y, body_z, 'k-', linewidth=2)[0]
        self.body_lines_3d.append(line)
        
        # 绘制机身中心点
        point = self.ax_3d.scatter([root_pos[0]], [root_pos[1]], [root_pos[2]], 
                                 c='black', s=50, marker='o')
        self.body_lines_3d.append(point)
        
    def draw_legs(self, root_pos, root_rot, dof_pos, foot_positions):
        """绘制机器人腿部"""
        # 清除之前的腿部线条
        for line in self.leg_lines_3d:
            line.remove()
        for point in self.foot_points_3d:
            point.remove()
        self.leg_lines_3d.clear()
        self.foot_points_3d.clear()
        
        # 髋关节位置
        L, W = self.go2.L, self.go2.W
        hip_positions = np.array([
            [L/2, -W/2, 0],   # FR
            [L/2, W/2, 0],    # FL
            [-L/2, -W/2, 0],  # HR
            [-L/2, W/2, 0]    # HL
        ])
        
        # 应用旋转和平移
        if np.linalg.norm(root_rot) > 1e-6:
            rotm = utils.quaternion2rotm(root_rot)
            hip_positions = (rotm @ hip_positions.T).T
        hip_positions += root_pos
        
        # 绘制腿部
        for i in range(4):
            # 髋关节到足端的线
            line = self.ax_3d.plot([hip_positions[i, 0], foot_positions[i, 0]],
                                 [hip_positions[i, 1], foot_positions[i, 1]],
                                 [hip_positions[i, 2], foot_positions[i, 2]],
                                 color=self.colors[i], linewidth=2, 
                                 label=f'{self.leg_names[i]} Leg')[0]
            self.leg_lines_3d.append(line)
            
            # 足端点
            point = self.ax_3d.scatter([foot_positions[i, 0]], 
                                     [foot_positions[i, 1]], 
                                     [foot_positions[i, 2]],
                                     c=self.colors[i], s=100, marker='o')
            self.foot_points_3d.append(point)
            
    def update_foot_trajectories(self, frame_idx):
        """更新足端轨迹显示"""
        # 清除之前的轨迹
        for line in self.foot_traj_lines:
            line.remove()
        for point in self.foot_current_points:
            point.remove()
        self.foot_traj_lines.clear()
        self.foot_current_points.clear()
        
        # 绘制到当前帧的轨迹
        if frame_idx > 0:
            for i in range(4):
                # 获取轨迹数据
                traj_x = []
                traj_y = []
                
                for f in range(frame_idx + 1):
                    frame_data = self.get_frame_data(f)
                    foot_pos = self.calculate_foot_positions(frame_data)
                    traj_x.append(foot_pos[i, 0])
                    traj_y.append(foot_pos[i, 1])
                
                # 绘制轨迹线
                line = self.ax_top.plot(traj_x, traj_y, color=self.colors[i], 
                                      linewidth=1, alpha=0.7, 
                                      label=f'{self.leg_names[i]} Trajectory')[0]
                self.foot_traj_lines.append(line)
                
                # 当前位置点
                if traj_x and traj_y:
                    point = self.ax_top.scatter([traj_x[-1]], [traj_y[-1]], 
                                              c=self.colors[i], s=100, marker='o')
                    self.foot_current_points.append(point)
                    
    def update_foot_heights(self, frame_idx):
        """更新足端高度显示"""
        # 清除之前的高度线
        for line in self.foot_height_lines:
            line.remove()
        self.foot_height_lines.clear()
        
        # 当前帧数据
        current_data = self.get_frame_data(frame_idx)
        foot_positions = self.calculate_foot_positions(current_data)
        
        # 绘制当前足端高度
        for i in range(4):
            y_pos = foot_positions[i, 1]
            z_pos = foot_positions[i, 2]
            
            point = self.ax_side.scatter([y_pos], [z_pos], c=self.colors[i], 
                                       s=100, marker='o', label=f'{self.leg_names[i]} Foot')
            self.foot_height_lines.append(point)
            
        # 地面线
        y_range = [-0.6, 0.6]
        ground_line = self.ax_side.plot(y_range, [0, 0], 'k--', linewidth=1, alpha=0.5)[0]
        self.foot_height_lines.append(ground_line)
        
    def update_com_trajectory(self, frame_idx):
        """更新质心轨迹显示"""
        # 清除之前的质心线
        for line in self.com_lines:
            line.remove()
        self.com_lines.clear()
        
        # 获取到当前帧的质心数据
        time_points = []
        x_positions = []
        y_positions = []
        z_positions = []
        
        for f in range(frame_idx + 1):
            frame_data = self.get_frame_data(f)
            time_points.append(f * self.frame_duration)
            x_positions.append(frame_data['root_pos'][0])
            y_positions.append(frame_data['root_pos'][1])
            z_positions.append(frame_data['root_pos'][2])
        
        # 绘制质心轨迹
        if time_points:
            x_line = self.ax_com.plot(time_points, x_positions, 'r-', 
                                    linewidth=2, label='X Position')[0]
            y_line = self.ax_com.plot(time_points, y_positions, 'g-', 
                                    linewidth=2, label='Y Position')[0]
            z_line = self.ax_com.plot(time_points, z_positions, 'b-', 
                                    linewidth=2, label='Z Position')[0]
            
            self.com_lines.extend([x_line, y_line, z_line])
            
            # 当前时刻的垂直线
            current_time = frame_idx * self.frame_duration
            y_min, y_max = self.ax_com.get_ylim()
            time_line = self.ax_com.axvline(x=current_time, color='black', 
                                          linestyle='--', alpha=0.7)[0]
            self.com_lines.append(time_line)
        
    def update_frame(self, frame_idx):
        """更新显示帧"""
        frame_idx = int(frame_idx)
        
        # 获取当前帧数据
        data = self.get_frame_data(frame_idx)
        foot_positions = self.calculate_foot_positions(data)
        
        # 更新3D视图
        self.ax_3d.clear()
        self.ax_3d.set_title(f'Robot Pose (Frame: {frame_idx}/{self.num_frames-1})')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        self.draw_robot_body(data['root_pos'], data['root_rot'])
        self.draw_legs(data['root_pos'], data['root_rot'], data['dof_pos'], foot_positions)
        
        # 设置3D视图范围
        center = data['root_pos']
        range_val = 0.5
        self.ax_3d.set_xlim([center[0]-range_val, center[0]+range_val])
        self.ax_3d.set_ylim([center[1]-range_val, center[1]+range_val])
        self.ax_3d.set_zlim([0, 0.5])
        
        # 更新其他视图
        self.update_foot_trajectories(frame_idx)
        self.update_foot_heights(frame_idx)
        self.update_com_trajectory(frame_idx)
        
        # 更新图例
        if frame_idx == 0:  # 只在第一帧添加图例
            self.ax_top.legend()
            self.ax_side.legend()
            self.ax_com.legend()
        
        # 刷新显示
        self.fig.canvas.draw()
        
    def show(self):
        """显示可视化窗口"""
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # 为滑条留出空间
        plt.show()


def main():
    """主函数"""
    import sys
    import os
    
    # 默认使用sidestep.json
    json_file = "output_json/sidestep.json"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"错误：文件 {json_file} 不存在")
        print("可用的轨迹文件：")
        if os.path.exists("output_json"):
            for f in os.listdir("output_json"):
                if f.endswith(".json"):
                    print(f"  - {f}")
        return
    
    # 创建可视化器
    print("创建轨迹可视化器...")
    visualizer = TrajectoryVisualizer(json_file)
    
    print("显示可视化窗口...")
    print("使用滑条可以控制播放进度")
    visualizer.show()


if __name__ == "__main__":
    main()