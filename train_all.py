import os

print("go2_dance_beat")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_beat --num_envs=53248 --headless")
print("go2_dance_turn_and_jump")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_turn_and_jump --num_envs=53248 --headless")
print("go2_dance_wave")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_wave --num_envs=53248 --headless")
print("go2_dance_pace")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_pace --num_envs=53248 --headless")
print("go2_dance_trot")
os.system("python legged_gym/legged_gym/scripts/train.py --task=go2_dance_trot --num_envs=53248 --headless")

# "python legged_gym/legged_gym/scripts/train_trans.py --task=go2_dance_trans --num_envs=53248 --headless"
# python legged_gym/legged_gym/scripts/train_panda.py --task=panda7_beat --num_envs=40960 --headless
# python legged_gym/legged_gym/scripts/train_panda.py --task=panda7_spacetrot --num_envs=4096 --headless

# panda7_fixed_arm
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_arm_beat --num_envs=40960 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_arm_turn_and_jump --num_envs=40960 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_arm_swing --num_envs=40960 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_arm_wave --num_envs=40960 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_arm_trot --num_envs=40960 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_arm_pace --num_envs=40960 --headless

# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_spacetrot --num_envs=4096 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_turn_and_jump --num_envs=4096 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_wave --num_envs=4096 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_trot --num_envs=4096 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_swing --num_envs=4096 --headless
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_stand --num_envs=4096 --headless --sim_device=cuda:1 --rl_device=cuda:1
# python legged_gym/legged_gym/scripts/train.py --task=panda7_fixed_gripper_arm_leg --num_envs=4096 --headless
