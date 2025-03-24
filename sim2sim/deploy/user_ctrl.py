import numpy as np
import torch

# model 0: stand
model_path_test0 = './model/go2/stand_2025-03-21_15-37-08.jit'  # 120ms延迟站立
# model 1: wave
model_path_test1 = './model/go2/wave_2025-03-24_09-39-30.jit'  #
# model 2: trot
model_path_test2 = './model/go2/stand_2025-03-21_15-37-08.jit'
# model 3: swing
model_path_test3 = './model/go2/swing_2025-03-24_08-54-00.jit'
# model 4: turn and jump
model_path_test4 = './model/go2/stand_2025-03-21_15-37-08.jit'  #
# model 5: wave two leg 1
model_path_test5 = './model/go2/stand_2025-03-21_15-37-08.jit'
# model 6: wave two leg 2
model_path_test6 = './model/go2/stand_2025-03-21_15-37-08.jit'

# 关节上下限，两前腿一致，两后腿一致
joint_up_limit = torch.tensor(
    [0.8378, 3.4907, -0.83776, 0.8378, 3.4907, -0.83776, 0.8378, 4.5379, -0.83776, 0.8378, 4.5379, -0.83776])
joint_low_limit = torch.tensor(
    [-0.8378, -1.5708, -2.7227, -0.8378, -1.5708, -2.7227, -0.8378, -0.5236, -2.7227, -0.8378, -0.5236, -2.7227])
# 更换腿部顺序
reindex_feet = [1, 0, 3, 2]

def quat_rotate_inverse(q, v):  # 获取基座z轴在惯性系下的投影矢量，q是IMU的四元数，v是z轴向量 (0,0,-1)
    """
    使用逆四元数旋转向量。

    参数:
        q (torch.Tensor): 四元数，形状为 [batch_size, 4]，格式为 [x, y, z, w]。
        v (torch.Tensor): 要旋转的向量，形状为 [batch_size, 3]。

    返回:
        torch.Tensor: 旋转后的向量，形状为 [batch_size, 3]。
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class userController:
    def __init__(self):
        self.device = torch.device('cpu')  # cpu cuda
        self.num_obs = 42  #
        self.num_actions = 12  # 12
        self.scale = {"lin_vel": 2.0,
                      "ang_vel": 0.25,
                      "dof_pos": 1.0,
                      "dof_vel": 0.05,  #0.05
                      "height_measurements": 5.0,
                      "clip_observations": 100.,
                      "clip_actions": 2.5,
                      "clip_arm_actions": 1.2,
                      "action_scale": 0.25}  #0.25
        default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]  # LF RF LH RH
        self.default_dof_pos = np.array(default_dof_pos)
        self.dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_actions, dtype=np.float32)

        self.actor_state = torch.zeros(size=(self.num_obs,), device=self.device, requires_grad=False)
        self.actions = torch.zeros(size=(self.num_actions,), device=self.device, requires_grad=False)

        self.p_gains = 20.0
        self.d_gains = 0.5
        self.dt = 0.02  #20ms
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.des_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)

        # saved data
        self.counter = 0
        self.saved_actor_state = []
        self.saved_torques = []
        self.saved_euler = []

        # model
        self.model_select = 0
        self.loadPolicy()

    def loadPolicy(self):
        # 测试模型，所有的都可以放进来
        self.model_test0 = torch.jit.load(model_path_test0).to(self.device)
        self.model_test0.eval()
        self.model_test1 = torch.jit.load(model_path_test1).to(self.device)
        self.model_test1.eval()
        self.model_test2 = torch.jit.load(model_path_test2).to(self.device)
        self.model_test2.eval()
        self.model_test3 = torch.jit.load(model_path_test3).to(self.device)
        self.model_test3.eval()
        self.model_test4 = torch.jit.load(model_path_test4).to(self.device)
        self.model_test4.eval()
        self.model_test5 = torch.jit.load(model_path_test5).to(self.device)
        self.model_test5.eval()
        self.model_test6 = torch.jit.load(model_path_test6).to(self.device)
        self.model_test6.eval()

    def PutToNet(self):
        x, y, z, w = self.quat[1], self.quat[2],self.quat[3], self.quat[0]
        base_quat = torch.tensor([x, y, z, w], dtype=torch.float32, device=self.device).unsqueeze(0)

        base_ang_vel_w = torch.from_numpy(self.ang_vel).unsqueeze(0)
        base_ang_vel = base_ang_vel_w.squeeze(0)

        gravity_vec = torch.tensor([0, 0, -1], dtype=torch.float32, device=self.device).unsqueeze(0)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec).squeeze(0)

        self.actor_state[0:3] = base_ang_vel * self.scale["ang_vel"]
        self.actor_state[3:6] = projected_gravity

        for i in range(4):  # LF RF LH RH <- RF LF RH LH
            for j in range(3):
                self.dof_pos[i * 3 + j] = self.qj[reindex_feet[i] * 3 + j]
                self.dof_vel[i * 3 + j] = self.dqj[reindex_feet[i] * 3 + j]

        self.actor_state[6: 6 + self.num_actions] = torch.from_numpy(self.dof_pos - self.default_dof_pos) * self.scale["dof_pos"]
        self.actor_state[6 + self.num_actions: 6 + self.num_actions * 2] = torch.from_numpy(self.dof_vel) * self.scale["dof_vel"]
        self.actor_state[6 + self.num_actions * 2: 6 + self.num_actions * 3] = self.actions

        self.actor_state = torch.clip(self.actor_state, -self.scale["clip_observations"],
                                      self.scale["clip_observations"]).to(self.device)

    def inference(self):
        self.PutToNet()

        # 将self.actor_state输出成文件
        self.saved_actor_state.append(self.actor_state.tolist())
        self.saved_euler.append(self.rpy)

        if self.counter == 3000:
            np.savetxt('data/actor_state.csv', np.array(self.saved_actor_state), delimiter=",")
            np.savetxt('data/base_euler.csv', np.array(self.saved_euler), delimiter=",")

            print('data has been saved')
        self.counter += 1

        with torch.no_grad():
            if self.model_select == 0:
                self.actions = self.model_test0(self.actor_state)
            if self.model_select == 1:
                self.actions = self.model_test1(self.actor_state)
            if self.model_select == 2:
                self.actions = self.model_test2(self.actor_state)
            if self.model_select == 3:
                self.actions = self.model_test3(self.actor_state)
            if self.model_select == 4:
                self.actions = self.model_test4(self.actor_state)
            if self.model_select == 5:
                self.actions = self.model_test5(self.actor_state)
            if self.model_select == 6:
                self.actions = self.model_test6(self.actor_state)


        # 剪切
        clip_actions = self.scale["clip_actions"] / self.scale["action_scale"]
        self.actions = torch.clip(self.actions, -clip_actions, clip_actions).to(self.device)

        actions_scaled = self.actions * self.scale["action_scale"]
        des_joint_pos = actions_scaled + self.default_dof_pos

        des_joint_pos = torch.clip(des_joint_pos, joint_low_limit, joint_up_limit)

        for i in range(4):   # RF LF RH LH  <- LF RF LH RH
            for j in range(3):
                self.des_joint_pos[i * 3 + j] = des_joint_pos[reindex_feet[i] * 3 + j]

        # print(f'RF hip error: {self.qj - self.des_joint_pos}')
