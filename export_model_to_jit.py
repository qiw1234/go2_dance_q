#!/usr/bin/env python3
"""
将训练好的 PyTorch 模型导出为 JIT 格式，用于在 raisim 中运行
"""

import os
import sys
import datetime
import copy

# 添加 legged_gym 路径
sys.path.append('/home/ubuntu/robot_dance/legged_gym')

# 先导入 isaacgym 相关模块
from legged_gym.envs import *
from legged_gym.utils import task_registry
from rsl_rl.modules import ActorCritic

# 最后导入 torch
import torch

def export_policy_as_jit(actor_critic, path, model_name="sidestep_model"):
    """
    导出策略为 JIT 格式
    """
    os.makedirs(path, exist_ok=True)
    
    # 获取当前日期和时间
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建文件名
    jit_name = f"{model_name}_{formatted_time}.jit"
    jit_path = os.path.join(path, jit_name)
    
    # 导出模型
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(jit_path)
    
    print(f"✅ 模型已导出到: {jit_path}")
    return jit_path

def load_and_export_model():
    """
    加载训练好的模型并导出为 JIT 格式
    """
    # 模型路径
    model_path = '/home/ubuntu/robot_dance/legged_gym/logs/go2_sidestep/run1/model_30000.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 获取任务配置
    task_name = "go2_sidestep"
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    
    print(f"📋 任务配置:")
    print(f"   - 任务名称: {task_name}")
    print(f"   - 观测维度: {env_cfg.env.num_observations}")
    print(f"   - 动作维度: {env_cfg.env.num_actions}")
    
    # 从checkpoint中获取实际的网络结构参数
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 从模型权重推断网络结构
    actor_input_dim = checkpoint['model_state_dict']['actor.0.weight'].shape[1]  # 42
    critic_input_dim = checkpoint['model_state_dict']['critic.0.weight'].shape[1]  # 96
    action_dim = checkpoint['model_state_dict']['actor.6.weight'].shape[0]  # 12
    
    print(f"📊 从模型推断的网络结构:")
    print(f"   - Actor 输入维度: {actor_input_dim}")
    print(f"   - Critic 输入维度: {critic_input_dim}")
    print(f"   - 动作维度: {action_dim}")
    
    # 创建策略网络（使用实际的维度）
    policy = ActorCritic(
        actor_input_dim,  # 42
        critic_input_dim,  # 96 (包含特权信息)
        action_dim,  # 12
        train_cfg.policy.actor_hidden_dims,
        train_cfg.policy.critic_hidden_dims,
        train_cfg.policy.activation,
        train_cfg.policy.init_noise_std
    ).to('cpu')
    
    # 加载训练好的权重
    print(f"📥 加载模型: {model_path}")
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✅ 模型加载成功!")
    print(f"   - 训练步数: {checkpoint.get('iter', 'Unknown')}")
    
    # 导出路径
    export_path = '/home/ubuntu/robot_dance/sim2sim/BJ_Raisim/net/HSW/model/go2'
    os.makedirs(export_path, exist_ok=True)
    
    # 导出为 JIT
    jit_path = export_policy_as_jit(policy, export_path, "go2_sidestep")
    
    return jit_path

def test_jit_model(jit_path):
    """
    测试导出的 JIT 模型
    """
    if not jit_path or not os.path.exists(jit_path):
        print("❌ JIT 模型不存在，跳过测试")
        return
    
    print(f"🧪 测试 JIT 模型: {jit_path}")
    
    try:
        # 加载 JIT 模型
        jit_model = torch.jit.load(jit_path)
        jit_model.eval()
        
        # 创建测试输入 (使用实际的观测维度)
        test_obs = torch.randn(1, 42)  # batch_size=1, obs_dim=42
        
        # 前向推理
        with torch.no_grad():
            action = jit_model(test_obs)
        
        print(f"✅ JIT 模型测试成功!")
        print(f"   - 输入维度: {test_obs.shape}")
        print(f"   - 输出维度: {action.shape}")
        print(f"   - 输出范围: [{action.min().item():.3f}, {action.max().item():.3f}]")
        
    except Exception as e:
        print(f"❌ JIT 模型测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始导出模型...")
    
    # 导出模型
    jit_path = load_and_export_model()
    
    # 测试模型
    test_jit_model(jit_path)
    
    if jit_path:
        print(f"\n🎉 导出完成!")
        print(f"📁 JIT 模型位置: {jit_path}")
        print(f"💡 现在您可以在 raisim 控制脚本中使用这个模型了")
        print(f"💡 需要修改 sim2sim/BJ_Raisim/net/HSW/bjtu_dance_twodogs_new_actions.py")
        print(f"💡 将您的模型路径添加到模型列表中")