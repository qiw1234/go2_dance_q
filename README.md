# 训练四足机器人跳舞
⚠️ 50系显卡没法用，isaac gym不再更新，因此新的显卡用不了。可以考虑isaac sim + isaac lab/genesis
## 1. 关于 
本分支的功能是训练四足机器人go2完成舞蹈动作，动作类型基本与主分支的panda7动作一致

⚠️ 如果分支内不是所有的文件都被git跟踪，最好不要在一个项目内添加多个分支，切换分支时有些文件没有跟踪可能会弄丢文件

## 2. 配置环境
1. 使用conda创建虚拟环境，python版本 3.6, 3.7 或 3.8 (推荐python3.8):
    - `conda create -n dance python==3.8`
    - `conda activate dance`
2. 安装 pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. 安装 Isaac Gym
   - 下载并安装 Isaac Gym Preview 4。网址： https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - 尝试例子测试是否安装成功 `cd examples && python 1080_balls_of_solitude.py`
   - 如果有问题查看文件路径： `isaacgym/docs/index.html`
4. 安装 rsl_rl (PPO implementation)
   -  `cd robot_dance/rsl_rl && pip install -e .` 
5. 安装 legged_gym
   - `cd ../legged_gym && pip install -e .`
6. 其余包
   - 缺什么安什么就可以，使用pip或者conda安装，可能会有版本不对应而报错
      ```bash
      conda install numpy=1.23
      pip install setuptools==59.5.0
      pip install protobuf==3.20.*
      ```
7. sim2sim
   - 进行sim2sim时，如果`cmake`会因为找不到头文件路径而报错，直接以项目形式打开`BJ_Raisim`文件夹
   - 安装raisim1.1.7，安装教程网址，搜索raisim：https://zhuanlan.zhihu.com/p/493505445
   - 注意安装的版本，不要直接从github上clone，找找release版本，raisim还在偷偷更新，很容易下错，出现运行速度慢的问题，具体表现是cpu单核占用接近100%
   - 根据raisim的路径修改`BJ_Raisim`中的相关的CMakeLists文件以及`run_panda7.sh`文件中相关的路径。
   - 用camke自动编译不报错即可

8. sim2real
   - sim2real是迁移到宇树的go2机器人，需要下载宇树的python_sdk2_python https://github.com/unitreerobotics/unitree_sdk2_python.git 随便找个地方放下载的文件即可，根据仓库readme下载并安装仓库。
   - 其仓库中提供了两种安装方式：pip安装以及源码安装，自选即可
   - 网线连接配置方法在宇树官网：https://support.unitree.com/home/zh/developer/Quick_start

## 3. 使用方法
1. 训练(train)
- train_all.py文件中有大部分动作的训练参数  
   ``` 
   python legged_gym/legged_gym/scripts/train.py --task=go2_swing --num_envs=4096 --headless --sim_device=cuda:1 --rl_device=cuda:1
   ```
   - --task 用于选择任务，具体的可选任务可以在`legged_gym/legged_gym/envs/__init__.py`中查看
   - --num_envs 用于设置环境数量，数量越多对显存需求越大
   - --headless 决定是否开启可视化窗口，没有这个参数默认开启
   - --sim_device 选择仿真设备
   - --rl_device 选择训练设备
   - --resume 是否从checkpoint恢复训练
   - --experiment_name 实验名字，用于保存checkpoint确定路径名称
   - --run_name 运行的名字，与实验名字作用类似
   - --max_iterations 最大迭代次数
   - 更多参数可以查看legged gym或者`legged_gym/legged_gym/utils/helpers.py`中`get_args`
2. 推理(play)
   ```
   python legged_gym/legged_gym/scripts/play.py --task=panda7_fixed_gripper_stand
   ```
   或者
   ```
   python legged_gym/legged_gym/scripts/play_panda.py --task=panda7_fixed_gripper_stand
   ```
   参数与训练类似

3. sim2sim  
   以项目形式打开`BJ_Raisim`文件夹
   
   新建一个终端，打开raisim仿真环境
   ```
   cd sim2sim/BJ_Raisim/dog_arm
   ./2run.sh
   ```
   - 如果`./2run.sh`运行不了，因为没有赋予权限
   - `sudo chmod +x ./2run.sh`即可  

   打开另一个终端，运行网络推理程序
   ```
   cd sim2sim/BJ_Raisim/net/HSW
   su
   conda activate your_envname
   python bjtu_dance_twodogs_new_actions.py
   ```
   进入管理员模式是为了使用按键控制，如果不用按键也可以不进入管理员模式
   - 如果`su`认证失败因为没有设置管理员密码
   - `su passwd`设置密码即可

4. sim2real
   - 实物实验是根据宇树官方的教程来完成的，可以通过网线连接机身，与其机器人内置电脑组成局域网，通过网线传输信息；也可以将程序部署到go2身上的nvidia nano主机上运行。控制方法的主要流程是利用pytorch加载出网络模型及参数，通过sdk获取网络输入信息，推理得到期望关节角度后向底层发布命令。
   - 注意网线连接方式，程序是在本地电脑上运行。
      ```
      python deploy_go2.py eth0
      ```
   - `eth0`表示网卡名字，需要根据自己的电脑网卡更换这个参数
   - 我们的程序在运行前会自动关闭宇树的运控服务，因此不用在app上手动关闭
   - 如果训练了新的网络，只需要添加对应的`user_ctrl_new`，代替原来的`user_ctrl`文件
   - 遥控器按键功能由`deploy_go2`中的`update_state_machine`方法以及`user_ctrl`中的`model_select`变量共同决定
   - 遥控器的对应变量可以查看宇树go2开发文档：https://support.unitree.com/home/zh/developer/Get_remote_control_status
   - 如果在go2上的机载电脑上运行，可以将`depoly`文件夹中的程序下载到狗上，在base环境中运行程序即可。程序一旦开始运行就不需要外部干预，可以拔掉hdmi等工具线。
   - 如果希望自启动，需要设置自启动脚本，在机载电脑开机后运行
      ```bash
      sudo nano /etc/systemd/system/service_name.service #myscript改成自己想要的名字
      ```
      文件中输入以下内容，部分内容需要修改
      ```
      [Unit]
      Description=My Startup Script
      After=network.target  # 确保网络就绪后执行，没有网卡也可以运行

      [Service]
      Type=simple
      User=username         # 指定运行用户 
      WorkingDirectory=/home/unitree/scripts/ #设置工作空间
      ExecStart=/path/to/your/script.sh
      Restart=on-failure    # 失败时自动重启
      RestartSec=5s         # 重启间隔时间

      [Install]
      WantedBy=multi-user.target
      ```
      检查文件是否有语法错误
      ```
      sudo systemd-analyze verify /etc/systemd/system/service_name.service
      ```
      设置权限，启动程序
      ```
      sudo systemctl daemon-reload         # 重载配置
      sudo systemctl enable myscript       # 设置自启动
      sudo systemctl start myscript        # 立即启动服务
      ```
      查看运行状态
      ```
      sudo systemctl status myscript       # 查看运行状态
      ```
      重启测试


      ⚠️如果要关闭开机自启动服务
      ```
      sudo systemctl disable myscript
      ```

   - 自动运行的脚本文件（.sh）可以按照如下方式，需要根据自己的功能做本地化修改
      ```
      #!/bin/bash

      # ========== Conda环境初始化 ==========
      # 加载Conda基础配置（解决"conda: command not found"问题）
      CONDA_BASE="/home/pcpc/anaconda3"  # 需替换为实际路径
      if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
         . "$CONDA_BASE/etc/profile.d/conda.sh"  # 加载环境变量
      else
         export PATH="$CONDA_BASE/bin:$PATH"      # 备用路径设置
      fi

      # 固定目录和程序名（按需修改）
      TARGET_DIR="deploy/real/go2"  # 👈 替换为你的目录路径
      PYTHON_SCRIPT="deploy_go2.py"        # 👈 替换为你的Python脚本名

      # 执行核心操作
      conda activate panda_walk
      cd "$TARGET_DIR" && python3 "$PYTHON_SCRIPT" "eno1"
      ```



## 4、文件结构
1. `legged_gym/legged_gym`中包含主要的环境文件；`envs`中包含每个任务的配置文件以及训练环境；`motion_loader`中包含参考轨迹的读取文件，可以根据特殊的要求进一步修改；`scripts`中包含训练以及推理的脚本文件

2. `opti_traj`中包含参考轨迹的生成文件以及保存好的参考轨迹

3. `rsl_rl`中包含ppo的实现方法，也可以根据需要修改

4. `sim2sim`中不仅包含sim2sim的内容，也包含了sim2real的内容
   - `BJ_Raisim`包含了raisim仿真的环境
   - `dog_arm`中的`2run.sh`,`run_panda7.sh`都可以打开raisim仿真环境，前者打开的是两个panda7机器人，后者打开的是一个。**而且如果`run_panda7.sh`报错，可以尝试运行`2run.sh`再运行`run_panda7.sh`**（程序有点bug）
   - `sim2sim/BJ_Raisim/dog_arm/user/DogSimRaisim2/CMakeLists.txt`中有可执行文件对应的源文件（.cpp）
   - `model`以及`nn_convert`都是采用非pytorch框架加载模型的方法，实际项目中没有使用
   - 其余文件都是绘图文件

5. sim2real
   - `deploy`中是迁移到go2实物机器人的文件
   - `analysis_pic`中保存了实物实验的图片
   - `data`中保存了实物实验的数据
   - `deploy_go2.py`是实物上运行的脚本文件
   - `plot.py`是实物实验绘图文件
   - `remote_controller.py`是遥控器类
   - `state_machine.py`是状态机
   - `test_user_ctrl.py`是测试用的用户控制器
   - `user_ctrl.py`是实际使用的用户控制器


### 参考：
- https://github.com/leggedrobotics/legged_gym
- https://github.com/inspirai/MetalHead
- https://github.com/Alescontrela/AMP_for_hardware
- https://support.unitree.com/home/zh/developer/Quick_start
- https://github.com/unitreerobotics/unitree_rl_gym


