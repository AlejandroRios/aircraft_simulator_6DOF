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
alpha_rad = 0.5;
beta_rad = -0.2;
C_ba = Cmat(2,alpha_rad)*Cmat(3,-beta_rad);
X = [C_ba*[500; 0; 0]
    0.7
    -0.8
    0.9
    1000
    900
    -10000
    -1
    1
    -1
    90];
U = [0.9
    20
    -15
    -20];

Xdot = dynamics(0,X,U);

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
H_ft_eq = 3000;
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

%% Linearização:
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
    Alin(:,j) = (Xdot_plus-Xdot_minus)/(2*h);
    Clin(:,j) = (Y_plus-Y_minus)/(2*h);
end

for j=1:length(U_eq)
    deltaU = zeros(length(U_eq),1);
    deltaU(j) = h;
    [Xdot_plus,Y_plus] = dynamics(0,X_eq,U_eq+deltaU);
    [Xdot_minus,Y_minus] = dynamics(0,X_eq,U_eq-deltaU);
    Blin(:,j) = (Xdot_plus-Xdot_minus)/(2*h);
    Dlin(:,j) = (Y_plus-Y_minus)/(2*h);
end

[eigenvec,eigenval] = eig(A);

diag(eigenval);
%% M ontagem de matrizes l
sel_lat=[2 6 10 4];
Alat = Alin(sel_lat,sel_lat);
Blat_a = Blin(sel_lat,3);
Blat_r = Blin(sel_lat,4);
Clat= Clin(:,sel_lat);
Dlat = Dlin(:,3);
Cbeta  = Clat(7,:);
Cphi   = Clat(8,:);
Cp     = Clat(9,:);
Cr     = Clat(10,:);
Hlat = [Clin(8,[2 6 10 4])*pi/180 D(1,3)
       Clin(10,[2 6 10 4]) D(1,3)];
Vtrim = 500;
%% Montagem de matrizes longitudinais para flare
Aa = [Alat Blat_a Blat_r zeros(4,2)
    zeros(2,4) [-20.2 0;0 -20.2] zeros(2,2)
    0 0 -1 0 zeros(1,4);
    0 1 0 0 0 0 0 -1];
rowNames = {'v','r','phi','p','d_a','d_r','Ephi','xw'};
colNames = {'v','r','phi','p','d_a','d_r','Ephi','xw'};
Aa_leg = array2table(Aa,'RowNames',rowNames,'VariableNames',colNames)
 
Ba = [zeros(4,2)
[20.2 0;0 20.2]
zeros(2,2)];
rowNames = {'v','r','phi','p','d_a','d_r','Epsi','xw'};
colNames = {'d_a','d_r'};
Ba_leg = array2table(Ba,'RowNames',rowNames,'VariableNames',colNames)

Ga = [zeros(6,2)
    1 0 
    0 0];

Ca = [zeros(1,6) 1 0
    -Cr 0 0 0 1
     Cp*pi/180 0 0 0 0
    -Cphi*pi/180 0 0 0 0];

rowNames = {'Ephi','er','p','ephi'};
colNames = {'v','r','phi','p','d_a','d_r','Ephi','xw'};
Ca_leg = array2table(Ca,'RowNames',rowNames,'VariableNames',colNames)

Fa = [0 0
       0 1 
       0 0
       1 0];
   
Ha = [Hlat [0 0 0;0 0 -1]];
rowNames = {'phi','rw'};
colNames = {'Beta','r','phi','p','d_a','d_r','Epsi','xw'};
Ha_leg = array2table(Ha,'RowNames',rowNames,'VariableNames',colNames) 
%% Realimentação com LGR da malha do nivelador de asas
k1_i = 0; % Ephi
k2_i = 0; % er
k3_i = 0; % p
k4_i = 0; % ephi 

K_i = [k1_i 0 k3_i k4_i
            0 k2_i 0 0];       
%% Realimentação de er
sys_2 = ss((Aa -Ba*K_i*Ca),Ba(:,2),Ca(2,:),0);
damp(sys_2)
% figure(1)
% rlocus(sys_2,-linspace(0,-40,2000))
% sgrid
k2_i = 0.18;
K_i = [k1_i 0 k3_i k4_i
         0 k2_i 0 0];
%% Realimentação de p 
sys_3 = ss((Aa -Ba*K_i*Ca),Ba(:,1),Ca(3,:),0);
% damp(sys_3)
% figure(2)
% rlocus(sys_3,-linspace(0,6,20000))
% sgrid
k3_i = -5;
K_i = [k1_i 0 k3_i k4_i
         0 k2_i 0 0];
%% Realimentação de erro phi   
sys_4 = ss((Aa -Ba*K_i*Ca),Ba(:,1),Ca(4,:),0);
damp(sys_4)
% figure(3)
% rlocus(sys_4,-linspace(0,-30,20000))
% sgrid
k4_i = 10;
K_i = [k1_i 0 k3_i k4_i
         0 k2_i 0 0];
%% Realimentação de integral erro phi   
sys_5 = ss((Aa -Ba*K_i*Ca),Ba(:,1),Ca(1,:),0);
damp(sys_5)
% figure(4)
% rlocus(sys_5,-linspace(0,-30,20000))
% sgrid
k1_i = 2;
K_i = [k1_i 0 k3_i k4_i
         0 k2_i 0 0];
%% Otimização dos ganhos
sys_fb= ss(Aa-Ba*K_i*Ca,Ba(:,1),Ca(3,:),0);
damp(sys_fb)
q = 0.01;
Q = Ha'*Ha;
R = [0.1 0;0 1];
r0 = [1;0];
V = 10;

opt = optimset('Display','iter','TolX',1e-12,'TolFun',1e-12);

% Obtendo a matriz de ganhos estabilizantes
K0 = [K_i(1,1),K_i(1,3),K_i(1,4),K_i(2,2)];
K1=fminsearch(@PI2,K0,opt,Aa,Ba,Ca,Fa,Ga,Ha,Q,R,r0,V,q);

K = [K1(1) 0 K1(2) K1(3)
      0   K1(4) 0 0];

% Fechamento de malha com ganhos otimizados nivelador de asas
Ac = Aa-Ba*K*Ca;
Bc = Ga-Ba*K*Fa;
Cc = [Ha; Ca];
sys_fb = ss(Ac,Bc,Ca,0);
damp(sys_fb)
% [Y,T,X] = step(sys_fb,40);
[Y,T,X] = initial(sys_fb,[0 0 10*pi/180 0 0 0 0 0],40);

figure(11)
subplot(3,2,1)
plot(T,X(:,1),'-b','linewidth',1)
grid on
ylabel('v [ft/s]')

subplot(3,2,2)
plot(T,(X(:,2)),'-b','linewidth',1)
grid on
ylabel('r [ft/s]')

subplot(3,2,3)
plot(T,(X(:,3)*180/pi),'-b','linewidth',1)
grid on
ylabel('\phi [deg]')

subplot(3,2,4)
plot(T,X(:,4),'-b','linewidth',1)
grid on
ylabel('p [deg/s]')

subplot(3,2,5)
plot(T,(X(:,5)),'-b','linewidth',1)
grid on
ylabel('d_a [deg]')
xlabel('T [s]')

subplot(3,2,6)
plot(T,(X(:,6)),'-b','linewidth',1)
grid on
ylabel('d_r [deg]')
xlabel('T [s]')

%%
fig11 = figure(11);
% export to pdf
set(fig11,'Units','Inches');
pos = get(fig11,'Position');
set(fig11,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig11,'fig11','-dpdf','-r0')

%% Adicionando dinamica do psi

[ns,~] = size(Ac);

A2 = [Ac, zeros(ns,1)
     0 1  zeros(1,ns-1)]
rowNames = {'v','r','phi','p','d_a','d_r','Ephi','xw','psi'};
colNames = {'v','r','phi','p','d_a','d_r','Ephi','xw','psi'};
A2_leg = array2table(A2,'RowNames',rowNames,'VariableNames',colNames)

B2 = [Bc(:,1)
      0];
rowNames = {'v','r','phi','p','d_a','d_r','Epsi','xw','psi'};
colNames = {'d_a'};
B2_leg = array2table(B2,'RowNames',rowNames,'VariableNames',colNames)
  
C2 = [zeros(1,ns) 1];
rowNames = {'psi'};
colNames = {'v','r','phi','p','d_a','d_r','Ephi','xw','psi'};
Ca_leg = array2table(C2,'RowNames',rowNames,'VariableNames',colNames)

D2 = 0;
%%
% Fechando a malha de psi mediante LGR
sys_psi=ss(A2,B2,C2,D2);
figure(2)
rlocus(sys_psi,-linspace(0,-20,2000))
k_psi = 3;
tau_psi = 500/(k_psi*9.8);
%%
Ac2 = A2-B2*k_psi*C2;
Bc2= B2*k_psi;
Cc2 = [Cc zeros(6,1)
    zeros(1,ns),1];
Dc2 = zeros(7,1);

sys2 = ss(Ac2,Bc2,Cc2,Dc2);

opt = stepDataOptions('InputOffset',0,'StepAmplitude',-5*pi/180);
[Y,T,X] = step(sys2,opt,50);
% [Y,T,X] = initial(sys2,[0 0 0 0 0 0 0 0 5*pi/180],100);

figure(13)
subplot(3,2,1)
plot(T,X(:,1),'-b','linewidth',1)
grid on
ylabel('v [ft/s]')

subplot(3,2,2)
plot(T,(X(:,2)),'-b','linewidth',1)
grid on
ylabel('r [deg/s]')

subplot(3,2,3)
plot(T,(X(:,3)),'-b','linewidth',1)
grid on
ylabel('\phi [deg]')


subplot(3,2,4)
plot(T,X(:,4),'-b','linewidth',1)
grid on
ylabel('p [deg/s]')

subplot(3,2,5)
plot(T,(X(:,5)),'-b','linewidth',1)
grid on
ylabel('d_a [deg]')
xlabel('T [s]')

subplot(3,2,6)
plot(T,(X(:,6)),'-b','linewidth',1)
grid on
ylabel('d_r [deg]')
xlabel('T [s]')
%%
fig13 = figure(13)
% % export to pdf
% set(fig13,'Units','Inches');
% pos = get(fig13,'Position');
% set(fig13,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% print(fig13,'fig13','-dpdf','-r0')


%%  
figure(14)
plot(T,(X(:,9)*180/pi),'-b','linewidth',1)
grid on
ylabel('\psi [deg]')
xlabel('T [s]')

%%
fig14 = figure(14);
set(fig14,'Units','Inches');
pos = get(fig14,'Position');
set(fig14,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig14,'fig14','-dpdf','-r0')


