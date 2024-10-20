import casadi as ca
import numpy as np

theta = ca.SX.sym('x')
rot_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -ca.sin(theta)], [0, ca.sin(theta), ca.cos(theta)]])
pos = [1, 2, 3]


# 3by3旋转矩阵变成4by4变换矩阵

temp = ca.SX.zeros(4,4)
a = rot_matrix
print(temp[2, 2])
print(rot_matrix)
temp[:3, :3] = rot_matrix
temp[2,2] = rot_matrix[2][2]
temp[:3,3] = pos
temp[3,:] = ca.SX([0, 0, 0, 1])

pos1 = [2, 3, 4]
print(ca.dot(pos1,pos1))

