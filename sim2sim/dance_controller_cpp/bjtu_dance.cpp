#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <unordered_map>

// 模型路径
std::string model_path_swing = "./model/swing/model_10000.jit";
std::string model_path_turnjump = "./model/turnjump/model_7450.jit";


class BJTUdance{
    public:
    BJTUdance();

    void on_key_press(const std::string& event);

    void listen_keyboard();

    void update_keyboard();

    void update_data();

    void loadPolicy();

    torch::Tensor compute_torques(torch::Tensor actions_scaled);

    void PutToDrive();

    void PutToNet();

    void PutToNet2();

    void inference_();

    private:
    int num_obs;
    int num_acts;
    std::unordered_map<std::string, int> scale;
    torch::Tensor default_dof_pos;
    torch::Tensor dof_pos;
    torch::Tensor dof_vel;
    torch::Tensor actor_state;
    torch::Tensor actions;
    torch::Tensor actor_state2;
    torch::Tensor actions2;
    torch::Tensor actions_last;
    torch::Tensor actions_last2;
    torch::Tensor p_gains, d_gains;
    torch::Tensor torques, torques2;
    torch::Tensor torque_limits;

    Eigen::MatrixXf joint_qd, joint_qd2;
    Eigen::MatrixXf joint_arm_d, joint_arm_d2;
    Eigen::MatrixXf foot_pos;
    float stand_height;

    int SHM_SIZE = 2*1024*1024;
    int SEM_KEY_ID = 0x5C0C0001;
};

BJTUdance::BJTUdance()
{
    num_obs = 60;
    num_acts = 18;
    scale["lin_vel"] = 2.0;
    scale["ang_vel"] = 0.25;
    scale["dof_pos"] = 1.0;
    scale["dof_vel"] = 0.05;
    scale["height_measurements"] = 5.0;
    scale["clip_observations"] = 100.0;
    scale["clip_actions"] = 2.5;
    scale["clip_arm_actions"] = 1.2;
    scale["action_scale"] = 0.25;
    default_dof_pos  = torch::tensor({0.1,0.8,-1.5,  -0.1,0.8,-1.5,  0.1,1.,-1.5,  -0.1,1.,-1.5, 0,0,0, 0,0,0}, torch::kCPU);
    dof_pos = torch::zeros({num_acts,1}, torch::kCPU);
    dof_vel = torch::zeros({num_acts,1}, torch::kCPU);

    actor_state = torch::zeros({num_obs, 1}, torch::kCPU);
    actions = torch::zeros({num_acts, 1}, torch::kCPU);

    actor_state2 = torch::zeros({num_obs, 1}, torch::kCPU);
    actions2 = torch::zeros({num_acts, 1}, torch::kCPU);

    actions_last = torch::zeros({num_acts, 1}, torch::kCPU);
    actions_last2 = torch::zeros({num_acts, 1}, torch::kCPU);

    p_gains = torch::tensor({150.,150.,150., 150.,150.,150.,  150.,150.,150.,  150.,150.,150.,  150.,150.,150., 20.,15.,10.});
    d_gains = torch::tensor({2.,2.,2.,  2.,2.,2.,  2.,2.,2.,  2.,2.,2.,  2.,2.,2., 0.1,0.1,0.1});

    torques = torch::zeros({num_acts, 1}, torch::kCPU);
    torques2 = torch::zeros({num_acts, 1}, torch::kCPU);

    torque_limits = torch::tensor({160,180,572,  160,180,572,  160,180,572,  160,180,572,  100,100,100, 100,100,100});

    joint_qd = Eigen::MatrixXf::Zero(4,3);
    joint_qd2 = Eigen::MatrixXf::Zero(4,3);
    joint_arm_d = Eigen::MatrixXf::Zero(num_acts-12,1);
    joint_arm_d2 = Eigen::MatrixXf::Zero(num_acts-12,1);

    foot_pos = Eigen::MatrixXf::Zero(4,3);
    stand_height = 0.52;

    // 加载模型
    loadPolicy();

}

void BJTUdance::loadPolicy()
{
    torch::jit::script::Module model_swing, model_turnjump;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model_swing = torch::jit::load(model_path_swing);
        model_turnjump = torch::jit::load(model_path_turnjump);
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    }
    model_swing.eval();
    model_turnjump.eval();
}

void BJTUdance::update_data(){
    
}


// 程序的主函数
int main()
{
    
    std::cout << "Length of line : " <<std::endl;
 
    return 0;
}
