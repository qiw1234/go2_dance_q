import casadi as ca
import numpy as np


def rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return e


def ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return e


def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    e = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return e


def rot2trans(rot_matrix, pos):
    '''3by3旋转矩阵变成4by4变换矩阵
    '''
    # temp = np.zeros([4, 4])
    # temp[:3, :3] = rot_matrix[:3,:3]
    # temp[:3, 3] = pos
    # temp[3, :] = np.array([0, 0, 0, 1])
    # return temp
    temp = ca.SX.eye(4)
    temp[:3,:3] = rot_matrix
    temp[:3,3] = pos
    # temp[3,:] = ca.SX([0, 0, 0, 1])
    return temp


def quaternion2rotm(quat):
    '''四元数变换成旋转矩阵

    Args:
        quat: x, y, z, w

    Returns:四元数对应的旋转矩阵

    '''
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]
    rotm = np.array([[1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]])
    return rotm


def quat2angvel_map(q):
    '''
    四元数的导数到角速度的映射矩阵
    Args:
        q: (x, y, z, w)

    Returns:
        返回映射矩阵，3*4，omega = 2*return @ q_dot(x,y,z,w)

    '''
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return np.array([[w, -z, y, -x], [z, w, -x, -y], [-y, x, w, -z]])


class QuadrupedRobot:
    def __init__(self):
        # go2模型参数
        self.L = 0.3868
        self.W = 0.093
        self.l1 = 0.0955
        self.l2 = 0.213
        self.l3 = 0.213
        self.eye = np.eye(3)
        self.p0 = np.array([0, 0, 0])
        self.toe =  np.array([self.l3, 0, 0, 1])
        self.lb = [-0.8378, -np.pi / 2, -2.7227, -0.8378, -np.pi / 2, -2.7227, -0.8378, -np.pi / 6, -2.7227, -0.8378,
              -np.pi / 6, -2.7227]
        self.ub = [0.8378, 3.4907, -0.8378, 0.8378, 3.4907, -0.8378, 0.8378, 4.5379, -0.8378, 0.8378, 4.5379, -0.8378]
        self.toe_pos_init = [ 0.178, -0.173, -0.3, 0.178, 0.173, -0.3, -0.178, -0.173, -0.3, -0.178, 0.173, -0.3]

    def rightfoot(self, q):
        "右侧腿末端到髋关节基坐标系的变换矩阵"
        trans01_right = rot2trans(rx(q[0]), self.p0) @ rot2trans(ry(np.pi / 2), self.p0)
        trans12_right = (rot2trans(rx(-np.pi / 2), self.p0) @
                         rot2trans(self.eye, np.array([0, 0, -self.l1])) @
                         rot2trans(rz(q[1]), self.p0))
        trans23_right = rot2trans(self.eye, np.array([self.l2, 0, 0])) @ rot2trans(rz(q[2]),
                                                                                         self.p0)
        return trans01_right @ trans12_right @ trans23_right

    def leftfoot(self, q):
        "左侧腿的末端到髋关节的基座标系的变换矩阵"
        trans01_left = rot2trans(rx(q[0]), self.p0) @ rot2trans(ry(np.pi / 2), self.p0)
        trans12_left = (rot2trans(rx(-np.pi / 2), self.p0) @
                        rot2trans(self.eye, np.array([0, 0, self.l1])) @
                        rot2trans(rz(q[1]), self.p0))
        trans23_left = rot2trans(self.eye, np.array([self.l2, 0, 0])) @ rot2trans(rz(q[2]),
                                                                                        self.p0)
        return trans01_left @ trans12_left @ trans23_left

    def trans(self, q, legnum):
        # "所有腿部的运动学变换矩阵,变换到形心处的机身坐标系下，就是还没考虑质心的位姿"
        # "legnum=0:FR  1:FL   2:HR    3:HL"
        # q表示关节角度
        if legnum == 0:
            trans_b0 = rot2trans(self.eye, np.array([self.L / 2, -self.W / 2, 0]))
            trans = trans_b0 @ self.rightfoot(q)
        elif legnum == 1:
            trans_b0 = rot2trans(self.eye, np.array([self.L / 2, self.W / 2, 0]))
            trans = trans_b0 @ self.leftfoot(q)
        elif legnum == 2:
            trans_b0 = rot2trans(self.eye, np.array([-self.L / 2, -self.W / 2, 0]))
            trans = trans_b0 @ self.rightfoot(q)
        elif legnum == 3:
            trans_b0 = rot2trans(self.eye, np.array([-self.L / 2, self.W / 2, 0]))
            trans = trans_b0 @ self.leftfoot(q)
        return trans

    def transrpy(self, q, legnum, rpy, p):
        '''
        考虑机身欧拉角的运动学模型
        legnum=0:FR  1:FL   2:HR    3:HL
        rpy=（r,p,y）,分别是绕x,y,z轴的转动角度
        p = (x,y,z) p表示形心的坐标3*1
        '''
        rotm = rz(rpy[2]) @ ry(rpy[1]) @ rx(rpy[0])
        transm = rot2trans(rotm, p)
        return transm @ self.trans(q, legnum)






# # 这是用来验证正运动学公式的
# go2 = QuadrupedRobot()
# toe_pos = np.array([go2.l3, 0, 0, 1])
# # dof_pos = np.array([0.8378, 2.48545, -2.18294])
#
# dof_pos = np.array([-0.1, 0.8, -1.5])
# #np中向量用一行表示
# root_rpy = np.array([0, 0, 0])
# root_pos = np.array([0, 0, 0.3])
# a = toe_pos.reshape(-1,1) # 求转置
# print(a.shape)  # (4, 1)
# print(toe_pos.shape)  # (4, )
# pos = go2.transrpy(dof_pos, 0, root_rpy, root_pos) @ toe_pos.reshape(-1, 1)
# print(pos)
# # quat = [0.2415334, 0.6404592, -0.5108982, 0.5200544 ]
# # rotm = quaternion2rotm(quat)
# # print(rotm)

