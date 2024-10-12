%考虑机身欧拉角的运动学模型
%legnum=0:FR  1:FL   2:HR    3:HL
%rpy=（r,p,y）,分别是绕x,y,z轴的转动角度
%p = (x,y,z) p表示形心的坐标3*1
function T = transrpy(q,legnum,robot,rpy,p)
        Rwb = rz(rpy(3))*ry(rpy(2))*rx(rpy(1));
        Twb = [Rwb,p;[0,0,0,1]];
        T = Twb*trans(q,legnum,robot);
end