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