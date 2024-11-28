clear all
close all
clc

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

%--------------------------------------------------------------------------
% Dados de inércia:
W = 55788*g;  
% Massa [slug]:
m = W/g;
% Momentos de inércia [slug·ft²]:
Ixx = 821466;
Iyy = 3343669;
Izz = 4056813;
% Produtos de inércia [slug·ft²]:
Ixy = 0;
Ixz = 178919;
Iyz = 0;
J = [Ixx -Ixy -Ixz
    -Ixy Iyy -Iyz
    -Ixz -Iyz Izz];
% CG nominal: 0.35c
xCG = 0;
yCG = 0;
zCG = 0;
rC_b = [xCG yCG zCG].';

%--------------------------------------------------------------------------
% Quantidade de movimento angular do motor [slug·ft²/s]: 
hex = 160;

r_pilot_b = [15; 0; 0];
aircraft = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'hex',hex,...
    'r_pilot_b',r_pilot_b);
%------------------------------------------------------
% Cálculo de equilibrio
xCG = 0;
yCG = 0;
zCG = 0;

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

% % Simulação da condição de equilibrio:
if trim_par.psi_dot_deg_s==0
tf = 20;
else
tf = 360/trim_par.psi_dot_deg_s;
end
tf = 10;
dt = 1e-2;

tic
options = odeset('MaxStep',dt);
% [Tsol,Xsol] = ode45(@dynamics,0:dt:tf,X_eq,[],U_eq);
[Tsol,Xsol] = ode45(@dynamics,0:dt:tf,X_eq,options,U_eq);
i_t = 1;
Usol = zeros(size(Xsol,1),length(U_eq));
Ysol = zeros(size(Xsol,1),length(Y_eq));
for u_t=0:dt:tf
    Usol(i_t,:) = U_eq.';
    [~,Y] = dynamics(u_t,Xsol(i_t,:).',U_eq);
    Ysol(i_t,:) = Y.';
    i_t = i_t+1;
end
time_ode45 = toc

plot_all_final
