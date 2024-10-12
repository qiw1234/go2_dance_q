%左侧腿的末端到髋关节的基座标系的变换矩阵
function T = leftfoot(q,robot)
        T01_left = R2T(rx(q(1)),robot.p0)*R2T(ry(pi/2),robot.p0);
        T12_left = R2T(rx(-pi/2),robot.p0)*R2T(robot.I,[0,0,robot.l1]')*R2T(rz(q(2)),robot.p0);
        T23_left = R2T(robot.I,[robot.l2,0,0]')*R2T(rz(q(3)),robot.p0);
        T = T01_left*T12_left*T23_left;
end