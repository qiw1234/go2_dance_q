clear,clc;
%定义运动参数
robot.L = 0.3868;
robot.W = 0.093;
robot.l1 = 0.0955;
robot.l2 = 0.213;
robot.l3 = 0.213;
robot.I = eye(3);
robot.p0 = [0 0 0]';
%q0：右前；q1：左前；q2:右后；q3:左后
%关节空间三个数代表机身关节、大腿关节、小腿关节
q0 = [-3.772059124660851e-04;1.370686308317024;-1.304782879323376];
q1 = [pi/8,pi/8,-pi/2];
q2 = [-pi/6,pi/4,-pi/2];
q3 = [0.3927,0.7854,-1.8326];
rpy = [0 0 0];
p = [0  0  0]';

%末端位置
P = [robot.l3,0,0,1]';
% P1_0 = trans(q0,0,robot)*P;
% p1_0 = P1_0(1:3);
% 
% P1_1 = trans(q1,1,robot)*P;
% p1_1 = P1_1(1:3);
% 
% P1_2 = trans(q2,2,robot)*P;
% p1_2 = P1_2(1:3);
% 
% P1_3 = trans(q3,3,robot)*P;
% p1_3 = P1_3(1:3);

%包含形心的欧拉角
P1_0 = transrpy(q0,0,robot,rpy,p)*P;
p1_0 = P1_0(1:3)




%旋转矩阵转换成缓缓矩阵，从3*3变到4*4
%输入的R是旋转矩阵，p是位置
function T = R2T(R,p)
        T = [R,p];
        T = [T;
            0,0,0,1];
end

%左侧腿的末端到髋关节的基座标系的变换矩阵
function T = leftfoot(q,robot)
        T01_left = R2T(rx(q(1)),robot.p0)*R2T(ry(pi/2),robot.p0);
        T12_left = R2T(rx(-pi/2),robot.p0)*R2T(robot.I,[0,0,robot.l1]')*R2T(rz(q(2)),robot.p0);
        T23_left = R2T(robot.I,[robot.l2,0,0]')*R2T(rz(q(3)),robot.p0);
        T = T01_left*T12_left*T23_left;
end

%右侧腿的末端到髋关节的基座标系的变换矩阵
function T = rightfoot(q,robot)
        T01_right = R2T(rx(q(1)),robot.p0)*R2T(ry(pi/2),robot.p0);
        T12_right = R2T(rx(-pi/2),robot.p0)*R2T(robot.I,[0,0,-robot.l1]')*R2T(rz(q(2)),robot.p0);
        T23_right = R2T(robot.I,[robot.l2,0,0]')*R2T(rz(q(3)),robot.p0);
        T = T01_right*T12_right*T23_right;
end

%所有腿部的运动学变换矩阵,变换到形心处的机身坐标系下，就是还没考虑质心的位姿
%legnum=0:FR  1:FL   2:HR    3:HL
function T = trans(q,legnum,robot)
        switch legnum
            case 0
                Tb0 = [robot.I,[robot.L/2,-robot.W/2,0]';[0,0,0,1]];
                T = Tb0*rightfoot(q,robot);
            case 1
                Tb0 = [robot.I,[robot.L/2,robot.W/2,0]';[0,0,0,1]];
                T = Tb0*leftfoot(q,robot);
            case 2 
                Tb0 = [robot.I,[-robot.L/2,-robot.W/2,0]';[0,0,0,1]];
                T = Tb0*rightfoot(q,robot);
            case 3
                Tb0 = [robot.I,[-robot.L/2,robot.W/2,0]';[0,0,0,1]];
                T = Tb0*leftfoot(q,robot);
        end
end

%考虑机身欧拉角的运动学模型
%legnum=0:FR  1:FL   2:HR    3:HL
%rpy=（r,p,y）,分别是绕x,y,z轴的转动角度
%p = (x,y,z) p表示形心的坐标3*1
function T = transrpy(q,legnum,robot,rpy,p)
        Rwb = rz(rpy(3))*ry(rpy(2))*rx(rpy(1));
        Twb = [Rwb,p;[0,0,0,1]];
        T = Twb*trans(q,legnum,robot);
end

