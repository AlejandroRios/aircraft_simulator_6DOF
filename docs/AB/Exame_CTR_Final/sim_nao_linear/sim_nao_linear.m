clear all
close all
clc

global g aircraft 
global m2ft ft2m kg2lb lb2kg kg2slug slug2kg deg2rad rad2deg

% O modelo do F-16 é construído no sistema de unidades inglesas:
m2ft = 1/0.3048;
ft2m = 1/m2ft;
g = 9.80665*m2ft;

% Outras conversões:
lb2kg = 0.45359237;
kg2lb = 1/lb2kg;

slug2kg = g*lb2kg;
kg2slug = 1/slug2kg;

deg2rad = pi/180;
rad2deg = 1/deg2rad;

%--------------------------------------------------------------------------
% Dados geométricos:
b = 30;     % Envergadura [ft]
S = 300;    % Área de referência [ft²]
c = 11.32;  % Corda média aerodinâmica [ft]

%--------------------------------------------------------------------------
% Dados de inércia:
W = 20500;
% Massa [slug]:
m = W/g;
% Momentos de inércia [slug·ft²]:
Ixx = 9496;
Iyy = 55814;
Izz = 63100;
% Produtos de inércia [slug·ft²]:
Ixy = 0;
Ixz = 982;
Iyz = 0;
J = [Ixx -Ixy -Ixz
    -Ixy Iyy -Iyz
    -Ixz -Iyz Izz];
% CG nominal: 0.35c
xCG = 0.0*c;
yCG = 0;
zCG = 0;
rC_b = [xCG yCG zCG].';

%--------------------------------------------------------------------------
% Quantidade de movimento angular do motor [slug·ft²/s]:
hex = 160;

r_pilot_b = [15; 0; 0];
aircraft = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'hex',hex,...
    'r_pilot_b',r_pilot_b);

% Testes: TABELA 3.5.2 Lewis
% alpha_rad = 0.5;
% beta_rad = -0.2;
% C_ba = Cmat(2,alpha_rad)*Cmat(3,-beta_rad);
% X = [C_ba*[500; 0; 0]
%     0.7
%     -0.8
%     0.9
%     1000
%     900
%     -10000
%     -1
%     1
%     -1
%     90];
% U = [0.9
%     20
%     -15
%     -20];
% 
% Xdot = dynamics(0,X,U);

%------------------------------------------------------
% Cálculo de equilibrio
xCG = 0.0*c;
yCG = 0;
zCG = 0;

rC_b = [xCG yCG zCG].';

r_pilot_b = [15; 0; 0];
aircraft = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'hex',hex,...
    'r_pilot_b',r_pilot_b);

V_eq = 500;
H_ft_eq = 200;
gamma_deg_eq = 0;
phi_dot_deg_s_eq = 0;
theta_dot_deg_s_eq = 0.*rad2deg;
psi_dot_deg_s_eq = 0*rad2deg;


trim_par = struct('V',V_eq,'H_ft',H_ft_eq,...
    'chi_deg',0,'gamma_deg',gamma_deg_eq,...
    'phi_dot_deg_s',phi_dot_deg_s_eq,...
    'theta_dot_deg_s',theta_dot_deg_s_eq,...
    'psi_dot_deg_s',psi_dot_deg_s_eq);

x_eq_0 = zeros(14,1);
x_eq_0(1) = V_eq;
options = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10);

[x_eq,fval,exitflag,output,jacobian] = fsolve(@trimF16,x_eq_0,options,trim_par);
while exitflag < 1
    [x_eq,fval,exitflag,output,jacobian] = fsolve(@trimF16,x_eq,options,trim_par);
end
X_eq = state_vec(x_eq,trim_par);
U_eq = control_vec(x_eq);
[Xdot_eq,Y_eq] = dynamics(0,X_eq,U_eq);

fprintf('----- TRIMMED FLIGHT PARAMETERS -----\n\n');
fprintf('   %-10s = %10.4f %-4s\n','x_CG',xCG,'ft');
fprintf('   %-10s = %10.4f %-4s\n','y_CG',yCG,'ft');
fprintf('   %-10s = %10.4f %-4s\n','z_CG',zCG,'ft');
fprintf('   %-10s = %10.4f %-4s\n','gamma',trim_par.gamma_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','chi',trim_par.chi_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi_dot',trim_par.phi_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta_dot',trim_par.theta_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi_dot',trim_par.psi_dot_deg_s,'deg/s');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','V',Y_eq(1),'ft/s');
fprintf('   %-10s = %10.4f %-4s\n','alpha',Y_eq(2),'deg');
fprintf('   %-10s = %10.4f %-4s\n','q',Y_eq(3),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta',Y_eq(4),'deg');
fprintf('   %-10s = %10.1f %-4s\n','H',Y_eq(5),'ft');
fprintf('   %-10s = %10.4f %-4s\n','beta',Y_eq(7),'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi',Y_eq(8),'deg');
fprintf('   %-10s = %10.4f %-4s\n','p',Y_eq(9),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','r',Y_eq(10),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi',Y_eq(11),'deg');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','throttle',100*U_eq(1),'%');
fprintf('   %-10s = %10.4f %-4s\n','delta_e',U_eq(2),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_a',U_eq(3),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_r',U_eq(4),'deg');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','n_x_pilot',Y_eq(13),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_pilot',Y_eq(14),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_pilot',Y_eq(15),'');
fprintf('   %-10s = %10.4f %-4s\n','n_x_CG',Y_eq(16),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_CG',Y_eq(17),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_CG',Y_eq(18),'');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','Mach',Y_eq(19),'');
fprintf('   %-10s = %10.2f %-4s\n','Dyn. p.',Y_eq(20),'lbf/ft^2');

if trim_par.psi_dot_deg_s==0
tf =50;
else
tf = 360/trim_par.psi_dot_deg_s;
end

dt = 1e-2;

Xinicial = [0 0 0 0 0 0 0 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]';

options = odeset('MaxStep',dt);
[Tsol,Xsol] = ode45(@feedback_app,0:dt:tf,(X_eq+Xinicial),options,U_eq,Y_eq);
i_t = 1;

Usol = zeros(size(Xsol,1),length(U_eq));
Ysol = zeros(size(Xsol,1),length(Y_eq));
Xdot_sol = zeros(size(Xsol,1),length(X_eq));
for u_t=0:dt:tf
   
   [Xdot,Y,U] = feedback_app(u_t,Xsol(i_t,:).',U_eq,Y_eq);
   Xdot_sol(i_t,:)  = Xdot.';
   Usol(i_t,:) = U.';
   Ysol(i_t,:) = Y.';
   i_t = i_t+1;
end

plot_all

