clear ;
close all;
clc;

%%
%添加casadi路径
addpath('D:\casadi');
import casadi.*;
%%
%添加物理参数
mass_val = 8.252;
Ib_val = [0.0576 0 -0.003;0 0.234 0;-0.003 0 0.2797];

%%
%腿接触地面的次序表参数
T = 0.5;%总时间
N = 16;
dt_val = repmat(T/(N-1),N-1,1);%间隔时间15*1
cs_val = [repmat([0 0 0 0]', 1, 2) repmat([1 1 1 1]', 1, 3) repmat([1 1 1 1]', 1, 10)];%4*15
cs_TD_val = zeros(4,N-1);%4*15

%%
%设置优化
% Optimization variables
% q       (6) [x,y,z,roll,pitch,yaw] (floating base state)
% qdot    (6) [vBody (world frame),omega (body frame)]实际上是反过来的
% c       (NLEGS*3) (absolute world frame)NLEGS表示腿数
% f_grf   (NLEGS*3) (absolute world frame)
%使用opti可以在casadi中快速创建NLP问题https://web.casadi.org/blog/opti/
%opti中有变量X,U，参数P，目标函数，约束，求解器，初步猜测值
opti = casadi.Opti();
X = opti.variable(12,N);
q = X(1:6,:);
q_dot = X(7:12,:);
U = opti.variable(24,N-1);
c = U(1:12,:);
f_grf = U(13:24,:);

%参数
X_ref = opti.parameter(12,N);
U_ref = opti.parameter(24,N-1);

cs = opti.parameter(4,N-1);    % contact schedule
cs_TD = opti.parameter(4,N-1); % contact schedule at touchdown
dt = opti.parameter(1,N-1);    % timesteps

q_min = opti.parameter(6,1);q_max = opti.parameter(6,1);
q_dot_min = opti.parameter(6,1);q_dot_max = opti.parameter(6,1);
q_init = opti.parameter(6,1);q_dot_init = opti.parameter(6,1);

c_init = opti.parameter(12,1);
q_term_min = opti.parameter(6,1);q_term_max = opti.parameter(6,1);%终止条件
q_dot_term_min = opti.parameter(6,1);q_dot_term_max = opti.parameter(6,1);

QX = opti.parameter(12,1);QN = opti.parameter(12,1);    % weighting matrices
Qc = opti.parameter(3,1);Qf = opti.parameter(3,1);

%机器人和环境参数
mu = opti.parameter();
l_leg_max = opti.parameter();
f_max = opti.parameter();
mass = opti.parameter();
Ib = opti.parameter(3,1);
Ib_inv = opti.parameter(3,1);
%g = opti.parameter();

p_foot_to_hip = [0.19;-0.1;-0.2;...
         0.19;0.1;-0.2;...
         -0.19;-0.1;-0.2;...
         -0.19;0.1;-0.2]; %
p_hip = [0.19,-0.1,0;...
         0.19,0.1,0;...
         -0.19,-0.1,0;...
         -0.19,0.1,0]; %之前这里有问题需要是一个矩阵，但是写成和上面一样的向量了

%%
%目标函数
cost = casadi.MX(0);
for i=1:N-1
    X_err = X(:,i)-X_ref(:,i);
    fp_err = repmat(X(1:3,i),4,1) + p_foot_to_hip - c(:,i);             %foot position error
    U_err = U(13:24,i)-U_ref(13:24,i);
    cost = cost + ((X_err)'*diag(QX)*(X_err)+...
           (fp_err)'*diag(repmat(Qc,4,1))*(fp_err)+...
           (U_err)'*diag(repmat(Qf,4,1))*(U_err));
end
XN_err = X(:,N) - X_ref(:,N);
cost = cost + XN_err'*(diag(QN))*XN_err;

opti.minimize(cost);

%%
%约束条件
%初始状态约束
opti.subject_to(q(1:6,1) == q_init);        % initial pos + ori
opti.subject_to(q_dot(1:6,1) == q_dot_init);    % initial ang. vel. + lin. vel.
% opti.subject_to(c(:,1) == c_init);          % initial foot positions


for k=1:N-1
%简化动力学约束
qk = q(:,k);             %6*1
qdk = q_dot(:,k);        %6*1[v on world frame;omega on body frame]
ck = c(:,k);             %12*1
fk = f_grf(:,k);         %12*1
rpyk = qk(4:6);
csk = cs(:,k);
%正常的body系转移到世界系应该是这样的，但是不知道为什么mit写出来的旋转矩阵
%rx，ry，rz和正常的是互为转置的，所以看起来很奇怪，把这个代码移动到别处是要注意
%我自己使用的旋转矩阵是正常的
R_body_to_world =rz(rpyk(3))*ry(rpyk(2))*rx(rpyk(1));
R_world_to_body = R_body_to_world';

% R_world_to_body = (rz(rpyk(3))' * ry(rpyk(2))' * rx(rpyk(1))')';
% R_body_to_world = rz(rpyk(3))' * ry(rpyk(2))' * rx(rpyk(1))';

vdot = 1/mass.*sum(reshape(fk,3,4),2)-[0;0;9.81];
omegadot = diag(Ib_inv)*(R_world_to_body*...
           (cross(ck(1:3)-qk(1:3),fk(1:3))+...
           cross(ck(4:6)-qk(1:3),fk(4:6))+...
           cross(ck(7:9)-qk(1:3),fk(7:9))+...
           cross(ck(10:12)-qk(1:3),fk(10:12)))-...
           cross(qdk(1:3),diag(Ib)*qdk(1:3)));

%这里第一个和第四个约束可能有问题，但是我不知道具体是什么问题，以后要是用到了再看有什么问题
%之前把总时间改成0.22s，然后把p_hip这个参数从12*1改成3*4还有问题，现在总时间改成0.5s就没问题了
opti.subject_to(q(1:3,k+1)-q(1:3,k) == qdk(4:6)*dt(k));
opti.subject_to(q(4:6,k+1)-q(4:6,k) == Binv(rpyk)*(R_body_to_world*qdk(1:3))*dt(k));
opti.subject_to(q_dot(1:3,k+1)-qdk(1:3) == omegadot*dt(k));
opti.subject_to(q_dot(4:6,k+1)-qdk(4:6) == vdot*dt(k));

%地面z向接触力不小于0
opti.subject_to(fk([3 6 9 12]) >= zeros(4,1));
%腾空时z向接触力为0，同时小于最大值
opti.subject_to(fk([3 6 9 12]) <= csk.*repmat(f_max,4,1));
%接触约束
for leg=1:4
    xyz_idx = 3*(leg-1)+1:3*(leg-1)+3;
    opti.subject_to(csk(leg)*ck(3*(leg-1)+3)==0);%接触时脚在地面上
    if (k+1<N)                                     %无滑动:接触地面时下一次脚的位置不变，不接触地面时下一次脚的位置可变
       stay_on_ground = repmat(csk(leg),3,1);
       opti.subject_to(stay_on_ground.*(c(xyz_idx,k+1)-c(xyz_idx,k))==0); 
    end
        r_hip = qk(1:3) + R_body_to_world*p_hip(leg,:)';
        p_rel = (ck(xyz_idx) - r_hip);
        kin_box_x = 0.05;
        kin_box_y = 0.05;
        kin_box_z = 0.27;
        
        opti.subject_to(-kin_box_x <= p_rel(1) <= kin_box_x);
        opti.subject_to(-kin_box_y <= p_rel(2) <= kin_box_y);
        opti.subject_to(-kin_box_z <= p_rel(3) + 0.05 <= 0);
        opti.subject_to(dot(p_rel, p_rel) <= l_leg_max^2);

end
    % friction Constraints, Eq (7k)
    opti.subject_to(fk([1 4 7 10]) <= 0.71*mu*fk([3 6 9 12]));
    opti.subject_to(fk([1 4 7 10]) >= -0.71*mu*fk([3 6 9 12]));
    opti.subject_to(fk([2 5 8 11]) <= 0.71*mu*fk([3 6 9 12]));
    opti.subject_to(fk([2 5 8 11]) >= -0.71*mu*fk([3 6 9 12]));
    
    % state & velocity bounds, Eq (7k)
    opti.subject_to(qk <= q_max);
    opti.subject_to(qk >= q_min);
    opti.subject_to(qdk <= q_dot_max);
    opti.subject_to(qdk >= q_dot_min);
    
end
%% reference trajectories
q_init_val = [0 0 0.35 0 0 0]';
qd_init_val = [0 0 0.0 0 0 -1]';

q_min_val = [-10 -10 -0 -10 -10 -10];
q_max_val = [10 10 0.4 10 10 10];
qd_min_val = [-10 -10 -10 -40 -40 -40];
qd_max_val = [10 10 10 40 40 40];

q_term_min_val = [-10 -10 0.15 -0.1 -0.1 -10];
q_term_max_val = [10 10 5 0.1 0.1 10];
qd_term_min_val = [-10 -10 -10 -40 -40 -40];
qd_term_max_val = [10 10 10 40 40 40];

q_term_ref = [0 0 0.2, 0 0 0]';
qd_term_ref = [0 0 0, 0 0 0]';

c_init_val = repmat(q_init_val(1:3),4,1)+...
    diag([1 -1 1, 1 1 1, -1 -1 1, -1 1 1])*repmat([0.2 0.1 -q_init_val(3)],1,4)';

c_ref = diag([1 -1 1, 1 1 1, -1 -1 1, -1 1 1])*repmat([0.2 0.1 -0.2],1,4)';
f_ref = zeros(12,1);

QX_val = [10 10 10, 10 10 10, 10 10 10, 10 10 10]';
QN_val = [0 0 100, 10 10 100, 10 10 10, 10 10 10]';
Qc_val = [0 0 0]';
Qf_val = [0.0001 0.0001 0.001]';

mu_val = 1;
l_leg_max_val = .3;
f_max_val = 200;
%% set parameter values
for i = 1:6
    Xref_val(i,:)   = linspace(q_init_val(i),q_term_ref(i),N);
    Xref_val(6+i,:) = linspace(qd_init_val(i),qd_term_ref(i),N);
end
for leg = 1:4
    for xyz = 1:3
        Uref_val(3*(leg-1)+xyz,:)    = Xref_val(xyz,1:end-1) + c_ref(3*(leg-1)+xyz);
        Uref_val(12+3*(leg-1)+xyz,:) = f_ref(xyz).*ones(1,N-1);
    end
end
opti.set_value(X_ref, Xref_val);
opti.set_value(U_ref, Uref_val);
opti.set_value(cs, cs_val);
opti.set_value(cs_TD, cs_TD_val);
opti.set_value(dt, dt_val);
opti.set_value(q_min, q_min_val);opti.set_value(q_max, q_max_val);
opti.set_value(q_dot_min, qd_min_val);opti.set_value(q_dot_max, qd_max_val);
opti.set_value(q_init, q_init_val);
opti.set_value(q_dot_init, qd_init_val);
opti.set_value(c_init, c_init_val);
opti.set_value(q_term_min, q_term_min_val);opti.set_value(q_term_max, q_term_max_val);
opti.set_value(q_dot_term_min, qd_term_min_val);opti.set_value(q_dot_term_max, qd_term_max_val);
opti.set_value(QX, QX_val);opti.set_value(QN, QN_val);
opti.set_value(Qc, Qc_val);opti.set_value(Qf, Qf_val);
opti.set_value(mu, mu_val);
opti.set_value(l_leg_max, l_leg_max_val);
opti.set_value(f_max,f_max_val);
opti.set_value(mass,mass_val);

opti.set_value(Ib,diag(Ib_val))
Ib_inv_val = diag(inv(Ib_val));
opti.set_value(Ib_inv,diag(inv(Ib_val)));

% opti.set_value(Ib,[0.576;0.234;0.2797]);
% opti.set_value(Ib_inv,[17.3775;4.2733;3.5776]);
%% initial guess
opti.set_initial([X(:);U(:)],[Xref_val(:);Uref_val(:)]);

%% casadi and IPOPT options
p_opts = struct('expand',true); % this speeds up ~x10
% experimental: discrete formulation of contact state
% p_opts.discrete = [zeros(12*N, 1);                  % floating base state, continuous
%                    zeros(6*model.NLEGS*(N-1),1);    % foot position + GRFS, continuous
%                    ones(model.NLEGS*(N-1), 1)];     % contact state, discrete


s_opts = struct('max_iter',3000,... %'max_cpu_time',9.0,...
    'tol', 1e-4,... % (1e-6), 1e-4 works well
    'acceptable_tol', 1e-4,... % (1e-4)
    'constr_viol_tol', 1e-3,... % (1e-6), 1e3 works well
    'acceptable_iter', 5,... % (15), % 5 works well
    'nlp_scaling_method','gradient-based',... {'gradient-based','none','equilibration-based'};
    'nlp_scaling_max_gradient',50,... % (100), % 50 works well
    'bound_relax_factor', 1e-6,... % (1e-8), % 1e-6 works well
    'fixed_variable_treatment','relax_bounds',... % {'make_parameter','make_constraint','relax_bounds'}; % relax bounds works well
    'bound_frac',5e-1,... % (1e-2), 5e-1 works well
    'bound_push',5e-1,... % (1e-2), 5e-1 works well
    'mu_strategy','adaptive',... % {'monotone','adaptive'}; % adaptive works very well
    'mu_oracle','probing',... % {'quality-function','probing','loqo'}; % probing works very well
    'fixed_mu_oracle','probing',... % {'average_compl','quality-function','probing','loqo'}; % probing decent
    'adaptive_mu_globalization','obj-constr-filter',... % {'obj-constr-filter','kkt-error','never-monotone-mode'};
    'mu_init',1e-1,... % [1e-1 1e-2 1]
    'alpha_for_y','bound-mult',... % {'primal','bound-mult','min','max','full','min-dual-infeas','safer-min-dual-infeas','primal-and-full'}; % primal or bound-mult seems best
    'alpha_for_y_tol',1e1,... % (1e1)
    'recalc_y','no',... % {'no','yes'};
    'max_soc',4,... % (4)
    'accept_every_trial_step','no',... % {'no','yes'}
    'linear_solver','mumps',... % {'ma27','mumps','ma57','ma77','ma86'} % ma57 seems to work well
    'linear_system_scaling','slack-based',... {'mc19','none','slack-based'}; % Slack-based
    'linear_scaling_on_demand','yes',... % {'yes','no'};
    'max_refinement_steps',10,... % (10)
    'min_refinement_steps',1,... % (1)
    'warm_start_init_point', 'no'); % (no)

s_opts.file_print_level = 0;
s_opts.print_level = 5;
s_opts.print_frequency_iter = 1;
s_opts.print_timing_statistics ='no';
opti.solver('ipopt',p_opts,s_opts);

%% solve
sol = opti.solve_limited();
X_star = sol.value(X);
U_star = sol.value(U);
plot3(X_star(1,:),X_star(2,:),X_star(3,:),'*',U_star(1,:),U_star(2,:),U_star(3,:),'O',U_star(4,:),U_star(5,:),U_star(6,:),'O',...
    U_star(7,:),U_star(8,:),U_star(9,:),'O',U_star(10,:),U_star(11,:),U_star(12,:),'O');










