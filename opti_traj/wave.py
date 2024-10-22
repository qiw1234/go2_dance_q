import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *

# 挥手动作：右前腿抬起来，挥舞右前腿，轨迹轮廓在y-z平面来看类似一片银杏叶
# 包含两段轨迹，一是抬腿到指定位置，不在一个平面内，需要通过两个平面图形进行组合，二是在yz平面内挥舞
def z_1(y, y0, z0):
    z = z0 - z0*(y + y0)**2/(y0**2)
    return z


go2 = utils.QuadrupedRobot()
num_row = 80
num_col = 49
fps = 50

turn_and_jump_ref = np.ones((num_row-1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row-1, 3))
root_ang_vel = np.zeros((num_row-1, 3))
root_rot_dot = np.zeros((num_row-1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row-1, 12))


