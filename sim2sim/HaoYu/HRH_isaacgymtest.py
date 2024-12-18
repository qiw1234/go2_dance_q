from isaacgym import gymtorch, gymapi
from isaacgym.terrain_utils import *
import time
import numpy as np
import torch
import os
from isaacgym.torch_utils import *
import threading
import keyboard
import csv
import math
file_path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/gym_60kg.csv'
joint_up_limit = [0.69,3.92,-0.52]
joint_low_limit = [-0.87,-1.46,-2.61]
file_0 = []
def s(x):
    return math.sin(x)
def c(x):
    return math.cos(x)
def t(x):
    return math.tan(x)

old = 0
if old == 1:
    new = 0
else:
    new = 1
max_effort = [160,180,572]
max_vel = [19.3,21.6,12.8]
class Configuration:
    class env:
        num_output = 12
        num_actor_input = 512
        if old == 1:
            num_sense_input = 420
        else:
            num_sense_input = 58*6
        num_sense_output = 16


    class net:
        dim_actor_latent = [256, 128, 64, 32]
        dim_sense_latent = [128, 64, 32]
        if old == 1:
            actor_domain = 70
        else:
            actor_domain = 58

    class model:
        # path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/70dog/model_200000.pt'
        # path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/70_new_urdf/model_85650.pt'
        # path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/50dog/model_200000.pt'
        # path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/70_new_urdf/model_100000.pt'
        # path = '/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/70HRH/model_53000.pt'
        # path ='/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/70HRH/model_HRH_404800.pt'
        path ='/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/70HRH/model_HRH_421700.pt'
        

class ActorNetwork(torch.nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.activation = torch.nn.ELU()
        self.cfg = Configuration()
        env_nums = 1
        '''线性变换层(全连接层)
        y=xA.T+bx x 是输入向量,A 是权重矩阵,b 是偏置向量

        torch.nn.Linear(in_features, out_features, bias=True)

            in_features: 输入的特征数，即每个输入样本的大小。
            out_features: 输出的特征数，即每个输出样本的大小。
            bias: 是否添加偏置项，默认为 True。
        '''

        self.actor_encoder1 = torch.nn.Linear(cfg.env.num_actor_input, cfg.net.dim_actor_latent[0])
        self.actor_encoder2 = torch.nn.Linear(cfg.net.dim_actor_latent[0], cfg.net.dim_actor_latent[1])
        self.actor_encoder3 = torch.nn.Linear(cfg.net.dim_actor_latent[1], cfg.net.dim_actor_latent[2])
        self.actor_encoder4 = torch.nn.Linear(cfg.net.dim_actor_latent[2], cfg.net.dim_actor_latent[3])

        self.sense_encoder1 = torch.nn.Linear(cfg.env.num_sense_input, cfg.net.dim_sense_latent[0])
        self.sense_encoder2 = torch.nn.Linear(cfg.net.dim_sense_latent[0], cfg.net.dim_sense_latent[1])
        self.sense_encoder3 = torch.nn.Linear(cfg.net.dim_sense_latent[1], cfg.net.dim_sense_latent[2])
        self.sense_encoder4 = torch.nn.Linear(cfg.net.dim_sense_latent[2], cfg.env.num_sense_output)

        self.decoder = torch.nn.Linear(in_features=cfg.net.dim_actor_latent[3], out_features=cfg.env.num_output)

    def forward(self, inputs):
        actor_desc = inputs['actor_desc']
        sense_desc = inputs['sense_desc']

        sense_x = self.activation(self.sense_encoder1(sense_desc))
        sense_x = self.activation(self.sense_encoder2(sense_x))
        sense_x = self.activation(self.sense_encoder3(sense_x))
        sense_x = self.sense_encoder4(sense_x)

        x = torch.cat((actor_desc[ :self.cfg.net.actor_domain], sense_x), dim=0)
        dim_nospace = self.cfg.net.actor_domain + self.cfg.env.num_sense_output
        x = torch.cat((x, actor_desc[ dim_nospace:]), dim=0)

        actor_x = self.activation(self.actor_encoder1(x))
        actor_x = self.activation(self.actor_encoder2(actor_x))
        actor_x = self.activation(self.actor_encoder3(actor_x))
        actor_x = self.activation(self.actor_encoder4(actor_x))

        return self.decoder(actor_x).unsqueeze(0)
    
class IsaacGYMTest:

    def __init__(self, urdf, num_envs = 1, device = 'cuda', drive_mode = ['torque', 'position']):
        self.drive_mode = drive_mode
        self.num_envs = num_envs
        self.device = device
        self.stand_height = 0.53
        self.x_vel = 0
        self.y_vel = 0
        self.load_mass = 0 #the mass of load
        self.dt = 1/50.
        self.kd = torch.tensor([19.3*0.1,21.6*0.1,12.8*0.1,19.3*0.1,21.6*0.1,12.8*0.1,19.3*0.1,21.6*0.1,12.8*0.1,19.3*0.1,21.6*0.1,12.8*0.1],device=self.device)
        self.kp = torch.tensor([160*0.8,180*0.8,572*0.8,160*0.8,180*0.8,572*0.8,160*0.8,180*0.8,572*0.8,160*0.8,180*0.8,572*0.8],device=self.device)
        self.joint_armatures = [0.035521313535,0.022005385155,0.12893591022]
        self.last_vel = None
        self.create_sim()
        self.add_ground()
        self.load_asset(urdf)
        self.create_env()
        self._init_buffer()
        self.event = threading.Event()
        self.key_pressed = None
        

    def create_sim(self,):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 0.005#0.005
        self.sim_params.substeps = 1
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.use_gpu_pipeline = True#force sensor 设置为false  hrh设置True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0
        self.sim_params.physx.default_buffer_size_multiplier = 5
        self.sim_params.physx.max_depenetration_velocity = 1.0
        self.sim_params.physx.max_gpu_contact_pairs = 2**23
        self.sim_params.physx.num_threads = 10
        self.sim_params.physx.bounce_threshold_velocity = 0.5 #2*9.81*self.sim_params.dt/self.sim_params.substeps
        self.sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
        self.sim_params.physx.use_gpu = True
        self.sim = self.gym.create_sim(0, 0,  gymapi.SIM_PHYSX, self.sim_params)

    def add_ground(self,):
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        self.plane_params.distance = 0
        self.plane_params.static_friction = 1
        self.plane_params.dynamic_friction = 1
        self.plane_params.restitution = 0
        self.gym.add_ground(self.sim, self.plane_params)
        num_terains = 1
        terrain_width = 10.
        terrain_length = 30.
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)
        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
        # heightfield[0:num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=4, step_height=0.4).height_field_raw#(new_sub_terrain(), step_width=0.3, step_height= 0.2).height_field_raw
        heightfield[0:num_rows, :] = sloped_terrain(new_sub_terrain(), slope=0.5).height_field_raw
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = 20
        tm_params.transform.p.y = -5.
        tm_params.transform.p.z = -0
        tm_params.transform.r.w =  0.707
        tm_params.transform.r.z =  0.707
        tm_params.transform.r.x =  0
        tm_params.transform.r.y =  0




        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def load_asset(self, urdf):
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.angular_damping = 0.0
        self.asset_options.armature = 0.0
        self.asset_options.collapse_fixed_joints = True
        self.asset_options.convex_decomposition_from_submeshes = False
        self.asset_options.density = 0.001
        self.asset_options.disable_gravity = False
        self.asset_options.fix_base_link = False
        self.asset_options.flip_visual_attachments = True
        self.asset_options.linear_damping = 0.0
        self.asset_options.max_angular_velocity = 1000.0
        self.asset_options.max_linear_velocity = 1000.0
        self.asset_options.replace_cylinder_with_capsule = True
        self.asset_options.thickness = 0.01

        if self.drive_mode == 'torque': self.asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        elif self.drive_mode == 'position': self.asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        else: raise ValueError

        self.asset = self.gym.load_asset(self.sim, os.path.dirname(urdf), os.path.basename(urdf), self.asset_options)


    def _init_buffer(self,): 
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(self.root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, 12, 2)
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]
        self.base_quat = self.root_state[..., 3:7]

        self.gravity_vec = to_torch(get_axis_params(-1, 2), device = self.device).repeat(self.num_envs, 1)

        #TODO: set commands, and commands_scales
        self.commands = torch.zeros((self.num_envs, 16), dtype = torch.float, device = self.device)
        self.actions = torch.zeros((self.num_envs, 12), dtype = torch.float, device = self.device)
        self.observation = torch.zeros((self.num_envs, 512), dtype = torch.float, device = self.device)
        self.commands_scales = torch.ones((self.num_envs, 16), dtype = torch.float, device = self.device)
        self.commands_scales[0,0] = 1
        self.commands_scales[0,1] = 0.25
        self.commands_scales[0,2] = 0.25
        self.commands_scales[0,3] = 0.05
        self.commands_scales[0,4] = 0
        self.commands_scales[0,5] = 1
        if old == 1 :
            self.ontology_sense_observation_buffer = torch.zeros((self.num_envs, 70, 6), dtype = torch.float, device = self.device)
        else:
            self.ontology_sense_observation_buffer = torch.zeros((self.num_envs, 58, 6), dtype = torch.float, device = self.device)

    def set_mass(self, props):
        '''loaded mass'''
        props[0].mass += self.load_mass

        return props

    def set_kpkd(self, props):
        #self.torque_limits = props['effort']
        if not hasattr(self, 'torque_limits'):
            self.torque_limits = torch.from_numpy(props['effort']).to(self.device)
            self.dof_pos_limits = torch.stack((torch.from_numpy(props['lower']), torch.from_numpy(props['upper'])), dim = 1).to(self.device)
            self.dof_vel_limits = torch.from_numpy(props['velocity']).to(self.device)
        if self.drive_mode == 'position':
            if old == 1:
                props['stiffness'].fill(100) #kp
                props['damping'].fill(1) #kd
            else:
                for i in range(4):
                    for j in range(3):
                        props['stiffness'][j+i*3]=(max_effort[j]*0.8) #kp
                        props['damping'][j+i*3]=(max_vel[j]*0.1) #kd
        for j in range(4):
            for i in range(3):
                props["armature"][j*3+i] = self.joint_armatures[i]
        return props
    
    def create_env(self,):
        self.num_envs = self.num_envs
        self.device = self.device
        envs_per_row = int(self.num_envs**0.5)
        env_spacing = 4
        env_lower = gymapi.Vec3(-env_spacing, 0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.envs = []
        env_actor_handles = {}
        self.actor_handles = []

        pose = gymapi.Transform()
        for env_index in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            self.envs.append(env)
            pose.p = gymapi.Vec3(2 * env_index, 0.0, 4)
            env_actor_handles[env_index] = self.gym.create_actor(env, self.asset, pose, str(env_index), env_index, env_index, 0)

            # self.gym.enable_actor_dof_force_sensors(self.envs[env_index], env_actor_handles[env_index])#advise to disable

            self.gym.set_actor_scale(env, env_actor_handles[env_index], 1)
            props = self.gym.get_actor_rigid_body_properties(env, env_actor_handles[env_index])
            self.gym.set_actor_rigid_body_properties(env, env_actor_handles[env_index], self.set_mass(props), recomputeInertia=True)
            props = self.gym.get_actor_dof_properties(env, env_actor_handles[env_index])
            self.gym.set_actor_dof_properties(env, env_actor_handles[env_index], self.set_kpkd(props))
            self.actor_handles.append(env_actor_handles)
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        self.gym.prepare_sim(self.sim)

    def on_key_press(self, event):
        self.key_pressed = event.name
        self.event.set()  # 设置事件，通知主线程处理

    def listen_keyboard(self):
        keyboard.on_press(self.on_key_press)
        while True:
            time.sleep(0.1)  # 模拟长时间运行

    def remapping(self, model):
        for n, p in self.net.named_parameters():
            if n == 'actor_encoder1.weight':
                p.data = model['actor.net.0.weight']
            if n == 'actor_encoder1.bias':
                p.data = model['actor.net.0.bias']
            if n == 'actor_encoder2.weight':
                p.data = model['actor.net.2.weight']
            if n == 'actor_encoder2.bias':
                p.data = model['actor.net.2.bias']
            if n == 'actor_encoder3.weight':
                p.data = model['actor.net.4.weight']
            if n == 'actor_encoder3.bias':
                p.data = model['actor.net.4.bias']
            if n == 'actor_encoder4.weight':
                p.data = model['actor.net.6.weight']
            if n == 'actor_encoder4.bias':
                p.data = model['actor.net.6.bias']
            if n == 'decoder.weight':
                p.data = model['actor.net.8.weight']
            if n == 'decoder.bias':
                p.data = model['actor.net.8.bias']
            if n == 'sense_encoder1.weight':
                p.data = model['actor.ontology_sense.net.0.weight']
            if n == 'sense_encoder1.bias':
                p.data = model['actor.ontology_sense.net.0.bias']
            if n == 'sense_encoder2.weight':
                p.data = model['actor.ontology_sense.net.2.weight']
            if n == 'sense_encoder2.bias':
                p.data = model['actor.ontology_sense.net.2.bias']
            if n == 'sense_encoder3.weight':
                p.data = model['actor.ontology_sense.net.4.weight']
            if n == 'sense_encoder3.bias':
                p.data = model['actor.ontology_sense.net.4.bias']
            if n == 'sense_encoder4.weight':
                p.data = model['actor.ontology_sense.net.6.weight']
            if n == 'sense_encoder4.bias':
                p.data = model['actor.ontology_sense.net.6.bias']

    def refresh_buffers(self,):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def step(self,):
        self.actions = self.net(self.compute_observation())
        self.target_position = self.dof_pos + self.actions
        for _ in range(4):
            if self.drive_mode == 'position':
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_position))
            if self.drive_mode == 'torque':
                torques = self.kp * (self.target_position - self.dof_pos) - self.kd * self.dof_vel
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques.clip(-self.torque_limits, self.torque_limits)))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            for ii in range(4):
                for jj in range(3):
                    if(self.dof_pos[0][3*ii+jj]>joint_up_limit[jj] or 
                                self.dof_pos[0][3*ii+jj]<joint_low_limit[jj]):
                                print(jj,"joint over limit :",self.dof_pos[0][3*ii+jj])
            # forces = []
            # for env_index in range(self.num_envs):
            #     forces = self.gym.get_actor_dof_forces(self.envs[env_index], self.actor_handles[env_index][env_index])
            # print(forces)
            
            writer = csv.writer(file_0)
            list = []
            list.append(self.root_state[:,7].tolist()[0])
            for i in range(4):
                for j in range(3):
                    list.append(self.target_position[0][i*3+j].tolist())     
            for i in range(4):
                for j in range(3):
                    list.append(self.dof_pos[0][i*3+j].tolist())
            for i in range(4):
                for j in range(3):
                    list.append(self.dof_vel[0][i*3+j].tolist())
            # for i in range(4):
            #     for j in range(3):
            #         list.append(torques[0][i*3+j].tolist())

            writer.writerow(list)##将该组数据记录为同一行

        print("vel : ",self.root_state[:,7:10].norm(dim=-1))


    def compute_observation(self,):
        self.refresh_buffers()
        projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_state[:, 10:13])
        a1 = self.dof_pos[0][1]
        a2 = self.dof_pos[0][2]
        l1 = 0.34
        l2 = 0.34
        A = np.array([[c(a1), -s(a1),0,l1*c(a1)],
                [s(a1), c(a1),0,l1*s(a1)],
                [0,0,1,0],
                [0,0,0,1]])
        B = np.array([[c(a2), -s(a2),0,l2*c(a2)],
                [s(a2), c(a2),0,l2*s(a2)],
                [0,0,1,0],
                [0,0,0,1]])
        # print("ik_height",(A@B)[0][3])
        #############################
        #TODO: check the scales
        #############################
        if self.event.is_set():
            if self.key_pressed == 'u':
                self.stand_height += 0.005
            if self.key_pressed =='i':
                self.stand_height -= 0.005
            if self.key_pressed =='w':
                self.x_vel += 0.05
            if self.key_pressed =='s':
                self.x_vel -= 0.05
            if self.key_pressed =='a':
                self.y_vel += 0.05
            if self.key_pressed =='d':
                self.y_vel -= 0.05
            self.event.clear()  # 重置事件，等待下一个按键
        if self.x_vel !=self.last_vel:
            print("vel = ",self.x_vel)
        self.last_vel = self.x_vel
        # print("stand_height",self.stand_height)
        self.commands[0,0] = self.stand_height#站立高度
        self.commands[0,1] = self.x_vel#x_vel
        self.commands[0,2] = self.y_vel#y_vel
        self.commands[0,3] = 0#yaw_vel
        self.commands[0,4] = 1
        self.commands[0,5] = 1
        self.commands[0,6] = 0

        if new == 1:
            buf = torch.cat((projected_gravity,
                                 torch.sin(self.dof_pos),
                                 torch.cos(self.dof_pos),
                                 base_ang_vel * 0.25,
                                 self.actions,
                                 self.commands * self.commands_scales,
                                 ), dim = -1)
            self.observation[:] = torch.cat((projected_gravity,
                                 torch.sin(self.dof_pos),
                                 torch.cos(self.dof_pos),
                                 base_ang_vel * 0.25,
                                 self.actions,
                                 self.commands * self.commands_scales,
                                 torch.zeros(self.num_envs, 454,device=self.device),
                                 ), dim = -1)
        else:
            buf = torch.cat((projected_gravity,
                                    self.dof_vel * 0.05,
                                    torch.sin(self.dof_pos) * 1,
                                    torch.cos(self.dof_pos) * 1,
                                    base_ang_vel * 0.25,
                                    self.actions,
                                    self.commands * self.commands_scales,
                                    ), dim = -1)
            self.observation[:] = torch.cat((projected_gravity,
                                 self.dof_vel * 0.05,
                                 torch.sin(self.dof_pos) * 1,
                                 torch.cos(self.dof_pos) * 1,
                                 base_ang_vel * 0.25,
                                 self.actions,
                                 self.commands * self.commands_scales,
                                 torch.zeros(self.num_envs, 442,device=self.device),
                                 ), dim = -1)
        
        
        self.ontology_sense_observation_buffer[:] = torch.cat([self.ontology_sense_observation_buffer[..., 1:], buf.unsqueeze(2)], dim = -1)
        return {'actor_desc': self.observation.view(-1), 'sense_desc': self.ontology_sense_observation_buffer.view(-1)}
    
    def get_policy(self,):
        '''load policy'''
        self.cfg = Configuration()
        self.net = ActorNetwork(self.cfg).to(self.device)
        self.model = torch.load(self.cfg.model.path,map_location=self.device)['model_state_dict']
        self.remapping(self.model)

    def inference(self,input_state,input_sense):
        inputs = {}
        inputs['actor_desc'] = input_state
        inputs['sense_desc'] = input_sense
        outputs = self.net(inputs)
        return outputs
        
        
        
if __name__ == '__main__':
    # uurdf = '/home/ixxuan/isaacgym_pre4/project/urdf/panda7-oleg-724/urdf/panda7-oleg-724.urdf'
    # uurdf = '/home/ixxuan/isaacgym_pre4/project/urdf/panda5/urdf/panda5.urdf'
    # uurdf = '/home/ixxuan/Project/ocs2/DogBrainCoreControl_blackdog_sim_panda/user/DogSimRaisim/rsc/panda7_oldleg_new/urdf/panda7.urdf'
    # uurdf ='/home/ixxuan/isaacgym_pre4/project/urdf/pandajm_dj/pandajm.urdf'
    # uurdf ='/home/ixxuan/isaacgym_pre4/project/urdf/pandajm_zc/pandajm.urdf'
    uurdf ='/home/ixxuan/isaacgym_pre4/project/urdf/aubo/urdf/panda7.urdf'
    # uurdf = '/home/ixxuan/isaacgym_pre4/project/urdf/panda7/urdf/panda7.urdf'
    
    test = IsaacGYMTest(urdf=uurdf,drive_mode='position')
    test.keyboard_thread = threading.Thread(target=test.listen_keyboard)
    test.keyboard_thread.start()
    test.get_policy()
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"The file {file_path} has been deleted.")
    with open('/home/ixxuan/Project/Tsinghua/Apr18_16-57-48_/gym_60kg.csv', 'a', newline='') as file_0:
        writer0 = csv.writer(file_0)
        list0=[]
        name=["joint_q_d","joint_q_a","joint_qd_a"]
        list0.append("body_x_vel")
        for n in name:
            for i in range(12):
                list0.append(f'{n}{i}')
        writer0.writerow(list0)
        while True:test.step()
    