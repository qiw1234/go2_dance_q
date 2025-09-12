#!/usr/bin/env python3
"""
安全测试您的侧步模型的脚本
"""

import numpy as np
import torch
import getsharememory_twodogs
import time

class SafeSidestepTester:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 加载您的模型
        self.sidestep_model = torch.jit.load('./model/go2/go2_sidestep_2025-09-11_23-02-45.jit').to(self.device)
        self.sidestep_model.eval()
        
        # 加载参考的站立模型进行对比
        self.stand_model = torch.jit.load('./model/go2/stand_2025-03-16_21-04-51.jit').to(self.device)
        self.stand_model.eval()
        
        # 初始化共享内存
        self.shareinfo_feed = getsharememory_twodogs.ShareInfo()
        self.shareinfo_send = getsharememory_twodogs.ShareInfo()
        
        # 创建共享内存
        self.shmaddr, self.semaphore = getsharememory_twodogs.CreatShareMem()
        
        # 不同的缩放参数进行测试
        self.test_scales = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        self.current_scale_idx = 0
        
        print("🔧 安全测试器初始化完成")
        print("📊 将测试不同的动作缩放参数:")
        for i, scale in enumerate(self.test_scales):
            print(f"   {i}: action_scale = {scale}")
        
    def get_observations(self):
        """获取观测数据"""
        # 获取机器人状态
        base_pos = np.array([
            self.shareinfo_feed.sensor_package.base_pos[0],
            self.shareinfo_feed.sensor_package.base_pos[1], 
            self.shareinfo_feed.sensor_package.base_pos[2]
        ])
        
        base_quat = np.array([
            self.shareinfo_feed.sensor_package.base_quat[0],
            self.shareinfo_feed.sensor_package.base_quat[1],
            self.shareinfo_feed.sensor_package.base_quat[2],
            self.shareinfo_feed.sensor_package.base_quat[3]
        ])
        
        base_lin_vel = np.array([
            self.shareinfo_feed.sensor_package.base_lin_vel[0],
            self.shareinfo_feed.sensor_package.base_lin_vel[1],
            self.shareinfo_feed.sensor_package.base_lin_vel[2]
        ])
        
        base_ang_vel = np.array([
            self.shareinfo_feed.sensor_package.base_ang_vel[0],
            self.shareinfo_feed.sensor_package.base_ang_vel[1],
            self.shareinfo_feed.sensor_package.base_ang_vel[2]
        ])
        
        # 关节位置和速度 (12个关节)
        dof_pos = np.array([self.shareinfo_feed.sensor_package.joint_pos[i] for i in range(12)])
        dof_vel = np.array([self.shareinfo_feed.sensor_package.joint_vel[i] for i in range(12)])
        
        # 构建观测向量 (总共42维)
        obs = np.concatenate([
            base_lin_vel,     # 3
            base_ang_vel,     # 3  
            base_quat,        # 4
            dof_pos,          # 12
            dof_vel,          # 12
            np.zeros(8)       # 8 (可能是历史动作或其他)
        ])
        
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def send_actions(self, actions, action_scale):
        """发送动作到机器人"""
        # 限制动作范围
        clip_actions = 2.5 / action_scale
        actions_clipped = torch.clip(actions, -clip_actions, clip_actions)
        
        # 缩放动作
        actions_scaled = actions_clipped * action_scale
        
        # 发送到机器人
        for i in range(12):
            self.shareinfo_send.servo_package.joint_pos[i] = float(actions_scaled[0, i])
        
        # 其他必要的设置
        self.shareinfo_send.servo_package.mode = 1
        self.shareinfo_send.servo_package.gait_id = 0
        self.shareinfo_send.servo_package.duration = 0.02
        
    def run_comparison_test(self):
        """运行对比测试"""
        print("\n🎯 开始对比测试...")
        print("⏱️  将运行5秒站立模型，然后5秒您的侧步模型")
        print("📈 观察机器人的稳定性和行为差异")
        
        start_time = time.time()
        test_duration = 10  # 总测试时间
        switch_time = 5     # 切换时间
        
        while time.time() - start_time < test_duration:
            current_time = time.time() - start_time
            
            # 获取观测
            obs = self.get_observations()
            
            with torch.no_grad():
                if current_time < switch_time:
                    # 前5秒使用站立模型
                    actions = self.stand_model(obs)
                    model_name = "站立模型"
                    action_scale = 0.25
                else:
                    # 后5秒使用您的侧步模型
                    actions = self.sidestep_model(obs)
                    model_name = "侧步模型"
                    action_scale = self.test_scales[self.current_scale_idx]
            
            # 发送动作
            self.send_actions(actions, action_scale)
            
            # 打印状态
            if int(current_time * 10) % 10 == 0:  # 每0.1秒打印一次
                print(f"⏰ {current_time:.1f}s - {model_name} (scale={action_scale}) - "
                      f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            
            time.sleep(0.02)  # 50Hz控制频率
        
        print("✅ 对比测试完成!")
        
    def run_scale_test(self):
        """测试不同缩放参数"""
        print(f"\n🔧 测试缩放参数: {self.test_scales[self.current_scale_idx]}")
        print("⏱️  将运行3秒，观察机器人行为")
        
        start_time = time.time()
        test_duration = 3
        action_scale = self.test_scales[self.current_scale_idx]
        
        while time.time() - start_time < test_duration:
            current_time = time.time() - start_time
            
            # 获取观测
            obs = self.get_observations()
            
            with torch.no_grad():
                actions = self.sidestep_model(obs)
            
            # 发送动作
            self.send_actions(actions, action_scale)
            
            # 打印状态
            if int(current_time * 10) % 5 == 0:  # 每0.5秒打印一次
                print(f"⏰ {current_time:.1f}s - scale={action_scale} - "
                      f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            
            time.sleep(0.02)
        
        print(f"✅ 缩放参数 {action_scale} 测试完成!")
        
        # 切换到下一个缩放参数
        self.current_scale_idx = (self.current_scale_idx + 1) % len(self.test_scales)

if __name__ == "__main__":
    tester = SafeSidestepTester()
    
    print("🚀 安全测试开始!")
    print("📋 测试选项:")
    print("   1. 对比测试 (站立模型 vs 侧步模型)")
    print("   2. 缩放参数测试")
    
    try:
        # 先运行对比测试
        tester.run_comparison_test()
        
        # 然后测试不同的缩放参数
        for _ in range(len(tester.test_scales)):
            input(f"\n⌨️  按回车键测试下一个缩放参数 ({tester.test_scales[tester.current_scale_idx]})...")
            tester.run_scale_test()
            
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        
    print("🏁 测试结束!")