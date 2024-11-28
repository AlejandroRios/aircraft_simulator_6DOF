% clear all
% close all
% clc

global g aircraft 
global m2ft ft2m kg2lb lb2kg kg2slug slug2kg deg2rad rad2deg

m2ft = 1/0.3048;
ft2m = 1/m2ft;
g = 9.80665;

% Outras conversões:
lb2kg = 0.45359237;
kg2lb = 1/lb2kg;

slug2kg = g*lb2kg;
kg2slug = 1/slug2kg;

deg2rad = pi/180;
rad2deg = 1/deg2rad;

%--------------------------------------------------------------------------
% Dados geométricos:
b = 32.757;     % Envergadura [m]
S = 116;    % Área de referência [m²]
c = 3.862;  % Corda média aerodinâmica [m]
% m = W/g;
%--------------------------------------------------------------------------
%% Dados de in?rcia: Conf. nominal
% % W = 55788*g;  
% % % Momentos de in?rcia [slug?ft?]:
% % Ixx = 821466;
% % Iyy = 3343669;
% % Izz = 4056813;
% % % Produtos de in?rcia [slug?ft?]:
% % Ixy = 0;
% % Ixz = 178919;
% % Iyz = 0;
%% Dados de in?rcia: Conf. 01
% W = 53620*g;  
% % Momentos de in?rcia [slug?ft?]:
% Ixx = 796100;
% Iyy = 3572000;
% Izz = 4262000;
% % Produtos de in?rcia [slug?ft?]:
% Ixy = 0;
% Ixz = 192300;
% Iyz = 0;
%% Dados de in?rcia: Conf. 02
% W = 53620*g;  
% % Momentos de in?rcia [slug?ft?]:
% Ixx = 796100;
% Iyy = 3501000;
% Izz = 4192000;
% % Produtos de in?rcia [slug?ft?]:
% Ixy = 0;
% Ixz = 177500;
% Iyz = 0;
%% Dados de in?rcia: Conf. 03
W = 53620*g;  
% Momentos de in?rcia [slug?ft?]:
Ixx = 796100;
Iyy = 3406000;
Izz = 4096000;
% Produtos de in?rcia [slug?ft?]:
Ixy = 0;
Ixz = 161200;
Iyz = 0;
%%

m = W/g;
J = [Ixx -Ixy -Ixz
    -Ixy Iyy -Iyz
    -Ixz -Iyz Izz];
%% CG nominal: 0.35c
% xCG = 0;
% yCG = 0;
% zCG = 0;
%% CG 01 
% xCG = 0.265;
% yCG = 0;
% zCG = -0.085;
%% CG 02
% xCG = -0.105;
% yCG = 0;
% zCG = -0.085;
%% CG 03
xCG = -0.515;
yCG = 0;
zCG = -0.085;
%%
rC_b = [xCG yCG zCG].';

%--------------------------------------------------------------------------
% Quantidade de movimento angular do motor [slug·ft²/s]: 
hex = 160;

r_pilot_b = [15; 0; 0];
aircraft = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'hex',hex,...
    'r_pilot_b',r_pilot_b);
%------------------------------------------------------
% Cálculo de equilibrio
% xCG = 0;
% yCG = 0;
% zCG = 0;

rC_b = [xCG yCG zCG].';

r_pilot_b = [15; 0; 0];
aircraft = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'hex',hex,...
    'r_pilot_b',r_pilot_b);

H_m_eq = 38000*ft2m;

[rho,~,~,a] = ISA(H_m_eq);

Mach = 0.78;
V_eq = Mach*a;

gamma_deg_eq = 0;
phi_dot_deg_s_eq = 0;
theta_dot_deg_s_eq = 0;
psi_dot_deg_s_eq = 0;
beta_deg_eq = 0;



trim_par = struct('V',V_eq,'H_m',H_m_eq,...
    'chi_deg',0,'gamma_deg',gamma_deg_eq,...
    'phi_dot_deg_s',phi_dot_deg_s_eq,...
    'theta_dot_deg_s',theta_dot_deg_s_eq,...
    'psi_dot_deg_s',psi_dot_deg_s_eq,...
    'beta_deg_eq',beta_deg_eq,...
        'W',[0;0;0]);

x_eq_0 = zeros(14,1);
x_eq_0(1) = V_eq;
options = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10);

[x_eq,fval,exitflag,output,jacobian] = fsolve(@trimGNBA,x_eq_0,options,trim_par);
while exitflag < 1
    [x_eq,fval,exitflag,output,jacobian] = fsolve(@trimGNBA,x_eq,options,trim_par);
end
X_eq = state_vec(x_eq,trim_par);
U_eq = control_vec(x_eq);
[Xdot_eq,Y_eq] = dynamics(0,X_eq,U_eq,trim_par.W);

fprintf('----- TRIMMED FLIGHT PARAMETERS -----\n\n');
fprintf('   %-10s = %10.4f %-4s\n','x_CG',xCG,'m');
fprintf('   %-10s = %10.4f %-4s\n','y_CG',yCG,'m');
fprintf('   %-10s = %10.4f %-4s\n','z_CG',zCG,'m');
fprintf('   %-10s = %10.4f %-4s\n','gamma',trim_par.gamma_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','chi',trim_par.chi_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi_dot',trim_par.phi_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta_dot',trim_par.theta_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi_dot',trim_par.psi_dot_deg_s,'deg/s');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','V',X_eq(1),'m/s');
fprintf('   %-10s = %10.4f %-4s\n','alpha',X_eq(2),'deg');
fprintf('   %-10s = %10.4f %-4s\n','q',X_eq(3),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta',X_eq(4),'deg');
fprintf('   %-10s = %10.1f %-4s\n','H',X_eq(5),'m');
fprintf('   %-10s = %10.4f %-4s\n','beta',X_eq(7),'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi',X_eq(8),'deg');
fprintf('   %-10s = %10.4f %-4s\n','p',X_eq(9),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','r',X_eq(10),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi',X_eq(11),'deg');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','Tle',U_eq(1),'N');
fprintf('   %-10s = %10.2f %-4s\n','Tre',U_eq(2),'N');
fprintf('   %-10s = %10.4f %-4s\n','ih',U_eq(3),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_e',U_eq(4),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_a',U_eq(5),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_r',U_eq(6),'deg');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','n_x_pilot',Y_eq(13),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_pilot',Y_eq(14),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_pilot',Y_eq(15),'');
fprintf('   %-10s = %10.4f %-4s\n','n_x_CG',Y_eq(16),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_CG',Y_eq(17),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_CG',Y_eq(18),'');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','Mach',Y_eq(19),'');
fprintf('   %-10s = %10.2f %-4s\n','Dyn. p.',Y_eq(20),'kg/ms^2');

    %% Linearizacao:

A = zeros(length(X_eq),length(X_eq));
B = zeros(length(X_eq),length(U_eq));

C = zeros(length(Y_eq),length(X_eq));
D = zeros(length(Y_eq),length(U_eq));

h = 1e-5;

for j=1:length(X_eq)
    deltaX = zeros(length(X_eq),1);
    deltaX(j) = h;
    [Xdot_plus,Y_plus] = dynamics(0,X_eq+deltaX,U_eq);
    [Xdot_minus,Y_minus] = dynamics(0,X_eq-deltaX,U_eq);
    A(:,j) = (Xdot_plus-Xdot_minus)/(2*h);
    C(:,j) = (Y_plus-Y_minus)/(2*h);
end

for j=1:length(U_eq)
    deltaU = zeros(length(U_eq),1);
    deltaU(j) = h;
    [Xdot_plus,Y_plus] = dynamics(0,X_eq,U_eq+deltaU);
    [Xdot_minus,Y_minus] = dynamics(0,X_eq,U_eq-deltaU);
    B(:,j) = (Xdot_plus-Xdot_minus)/(2*h);
    D(:,j) = (Y_plus-Y_minus)/(2*h);
end

[eigenvec,eigenval] = eig(A);

diag(eigenval)



% diag(eigenval)
sel_long=[1 2 3 4 5];
% sel_long=[2 3];
sel_lat=[7 8 9 10];
% realimentando q
% % % apenas mov long
eig(A(sel_long,sel_long))
Along = A(sel_long,sel_long);
Blong = B(sel_long,3);
Clong = C(:,sel_long);
Dlong = D(:,3);

sys_pc =ss(Along,Blong,Clong,Dlong);
damp(sys_pc)
% pzmap(sys_pc)
% % opt = stepDataOptions('InputOffset',0,'StepAmplitude',1);
% % [Y1,T1,X1]=step(sys_pc,opt);
% [Y1,T1,X1]=initial(sys_pc,[0 1 0 0 0 0 0 0 0],2500);
% figure(1) 
% subplot(611)
% plot(T1,X1(:,1),'k--','linewidth',1)
% ylabel('V [m/s]')
% grid on
% hold all
% subplot(612)
% plot(T1,X1(:,2),'k--','linewidth',1)
% ylabel('\alpha [deg]')
% grid on
% hold all
% subplot(613)
% plot(T1,X1(:,3),'k--','linewidth',1)
% ylabel('q [deg/s]')
% grid on
% hold all
% subplot(614)
% plot(T1,X1(:,4),'k--','linewidth',1)
% ylabel('\theta [deg]')
% grid on
% hold all
% subplot(615)
% plot(T1,X1(:,5),'k--','linewidth',1)
% ylabel('H [m]')
% grid on
% hold all
% subplot(616)
% plot(T1,X1(:,5),'k--','linewidth',1)
% ylabel('H [m]')
% grid on
% hold all


% %%
% fig1 = figure(1)
% % export to pdf
% set(fig1,'Units','Inches');
% pos = get(fig1,'Position');
% set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig1,'Ex_10_2','-dpdf','-r0')
