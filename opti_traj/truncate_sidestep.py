import json
import numpy as np

# 读取原始的sidestep轨迹文件
with open('opti_traj/output_json/sidestep.json', 'r') as f:
    original_data = json.load(f)

# 截取前79帧
truncated_frames = original_data['frames'][:79]

# 创建新的轨迹数据
truncated_data = {
    'frame_duration': original_data['frame_duration'],
    'frames': truncated_frames
}

# 保存截取后的轨迹
with open('opti_traj/output_json/sidestep_truncated.json', 'w') as f:
    json.dump(truncated_data, f, indent=4)

print(f"原始轨迹帧数: {len(original_data['frames'])}")
print(f"截取后帧数: {len(truncated_frames)}")
print(f"截取后持续时间: {len(truncated_frames) * original_data['frame_duration']:.2f}秒")
print(f"截取后的轨迹已保存到: opti_traj/output_json/sidestep_truncated.json")

# 同时保存txt格式
truncated_array = np.array(truncated_frames)
np.savetxt('opti_traj/output/sidestep_truncated.txt', truncated_array, delimiter=',')
print(f"同时保存txt格式到: opti_traj/output/sidestep_truncated.txt")