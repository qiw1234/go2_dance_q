clear,clc;
N = 500;
h = 0.2;
a1 = 0.02;%向后伸的最大深度
a2 = 0.02;%向前伸的最大深度
tstart = 0;
tend = 5;

T = 1;
z_s = [];
x_s = [];
for t = linspace(tstart,tend,N)
    n = fix(( t - tstart ) / T);%fix函数向0的方向取整
    z = -4*h*(t-(2*n+1)*T/2)^2+h;
    if sin(2*pi*t/T)>0
        x = -a1*sin(2*pi*t/T);
    else 
        x = -a2*sin(2*pi*t/T);
    end
    z_s = [z_s;z];
    x_s = [x_s;x];
end

plot(x_s,z_s, 'linewidth', 2)
axis equal%这个要把图话好之后再用才能生效
xlabel('X-axis');
ylabel('Z-axis');