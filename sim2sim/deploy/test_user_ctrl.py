import numpy as np


class userController:
    def __init__(self):
        self.num_actions = 12
        self.p_gains = 60.0
        self.d_gains = 5.0
        self.dt = 0.002
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.des_joint_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        self._targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]

        self.startPos = [0.0] * 12
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0

        self.firstRun = True

    def inference(self):
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.qj[i]
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.des_joint_pos[i] = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]

        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.des_joint_pos[i] = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(12):
                self.des_joint_pos[i] = self._targetPos_2[i]

        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(12):
                self.des_joint_pos[i] = (1 - self.percent_4) * self._targetPos_2[i] + self.percent_4 * self._targetPos_3[i]
        # print(f'RF hip error: {self.qj - self.des_joint_pos}')
