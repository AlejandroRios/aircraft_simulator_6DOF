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
%% M ontagem de matrizes longitudinais
sel_long=[1 3 5 11 13];

Along = Alin(sel_long,sel_long);
Blong_p = Blin(sel_long,2);
Blong_t = Blin(sel_long,1);
Clong = Clin(:,sel_long);
Dlong = Dlin(:,2);
Calpha = Clong(2,:);
Cq    = Clong(3,:);
Ctheta    = Clong(4,:);
Hlong = [Clin(1,[1,3,5,11,13]) D(1,2);];
Vtrim = 500;
%% Montagem de matrizes longitudinais para flare
A = [Along Blong_p zeros(5,1); 
     zeros(1,5) -20.2 0;
     Vtrim*(Ctheta-Calpha) zeros(1,2)];
 
B = [Blong_t zeros(5,1); 0 20.2; 0 0];

Bgamma = [zeros(6,1);-Vtrim];
Bagamma = [Bgamma;zeros(2,1)];

C = [Calpha 0 0;Ctheta 0 0;Cq 0 0];
G = eye(2);
F = zeros(2);
D = [0,0;1,0;0,0;0,1];
J = [1,0;0,0;0,1;0,0];
H = [Hlong 0;
    zeros(1,6) 1];
%% Montagem de matizes aumentadas
Aa = [A  zeros(7,2);
    -G*H,F];       % dinamica do compensador
rowNames = {'u','w','q','theta','Pow','d_p','d','Ev','Ed'};
colNames = {'u','w','q','theta','Pow','d_p','d','Ev','Ed'};
A_label = array2table(Aa,'RowNames',rowNames,'VariableNames',colNames)

Ba = [B;zeros(2,2)];
rowNames = {'u','w','q','theta','Pow','d_p','d','ev','ed'};
colNames = {'d_t','d_e'};
B_label = array2table(Ba,'RowNames',rowNames,'VariableNames',colNames)

Ca = [C zeros(3,2);
     -J*H,D]; % saida do compensador
rowNames = {'alpha','theta','q','ev','intev','ed','inted'};
colNames = {'u','w','q','theta','Pow','d_p','d','ev','ed'};
C_label = array2table(Ca,'RowNames',rowNames,'VariableNames',colNames)
 
Ha = [H zeros(2,2)];
rowNames = {'V','d'};
colNames = {'u','w','q','theta','Pow','d_p','d','ev','ed'};
H_label = array2table(Ha,'RowNames',rowNames,'VariableNames',colNames)

Fa = [zeros(3,2);
    J];

Ga = [zeros(7,2);
    G];
%% Realimentação com LGR 
ka_ini = 0;    
kt_ini  = 0;
kq_ini = 0;
kev_ini = 0;
kintev_ini =0;
ked_ini = 0;
kinted_ini =0;

k1 = 0;     % alpha
k2 = 0;     % theta
k3 = 0;     % q
k4 = 0;     % ev
k5 = 0;     % Ev
k6 = 0;     % ed
k7 = 0;     % Ed

k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];
%% Realimentação de alpha
sys_a = ss(Aa-Ba*k_ini*Ca,Ba(:,2),Ca(1,:),0);
damp(sys_a);
% figure(1)
% rlocus(sys_a,-linspace(0,5,20000))
k1 = -2;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];
%% Realimentação de theta
sys_t = ss(Aa-Ba*k_ini*Ca,Ba(:,2),Ca(2,:),0);
damp(sys_t);
% figure(2)
% rlocus(sys_t,-linspace(0,1,20000))
k2 = -0.2;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7]; 
%% Realimentação de q
sys_q = ss(Aa-Ba*k_ini*Ca,Ba(:,2),Ca(3,:),0);
damp(sys_q);
% figure(3)
% rlocus(sys_q,-linspace(0,10,20000))
k3 = -0.723;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];   
%% Realimentação de ed
sys_ed = ss(Aa-Ba*k_ini*Ca,Ba(:,2),Ca(6,:),0);
damp(sys_ed);
% figure(4)
% rlocus(sys_ed,-linspace(0,-0.1,20000))
k6 = 0.00022;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];
%% Realimentação de int ed
sys_inted = ss(Aa-Ba*k_ini*Ca,Ba(:,2),Ca(7,:),0);
damp(sys_inted);
% figure(4)
% rlocus(sys_inted,-linspace(0,-0.01,20000))
k7 = 6e-6;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];
%% Realimentação de ev
sys_ev = ss(Aa-Ba*k_ini*Ca,Ba(:,1),Ca(4,:),0);
damp(sys_ev);
% figure(4)
% rlocus(sys_ev,-linspace(0,0.1,20000))
k4 = -0.0023;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];
%% Realimentação de int ev
sys_intev = ss(Aa-Ba*k_ini*Ca,Ba(:,1),Ca(5,:),0);
damp(sys_intev);
% figure(4)
% rlocus(sys_intev,-linspace(0,0.0001,20000))
k5 = -1e-6;
k_ini = [0 0  0  k4  k5  0  0 ;
        k1 k2 k3 0   0   k6 k7];
%% Verificação dos polos do sistema realimentado
sys_fb = ss(Aa-Ba*k_ini*Ca,Ba,Ca(3,:),0);
damp(sys_fb);
%% Fechamento de malha com ganhos de LGR
K1 = k_ini;
Ac = Aa-Ba*K1*Ca;
Bc = Ga-Ba*K1*Fa;
Cc = Ca;
Dc = [0 0; 0 0; 0 0;D];
Sys = ss(Ac,Bc,Cc,Dc);
damp(Sys)
%% Otimização dos ganhos
Q = Ha'*Ha;
q = 0.001;
R =[0.01,0;0,0.001];
V = 0;
r0 = [1;1];

opt = optimset('TolX',1e-20,'TolFun',1e-20);
K0 = [k_ini(1,4),k_ini(1,5),k_ini(2,1),k_ini(2,2),k_ini(2,3),k_ini(2,6),k_ini(2,7)];
K1=fminsearch(@PI2,K0,opt,Aa,Ba,Ca,Fa,Ga,Ha,Q,R,r0,V,q);

K = [0,0,0,K1(1),K1(2), 0,0;
    K1(3),K1(4),K1(5),0,0 K1(6),K1(7)];

% Fechamento de malha com ganhos otimizados e inclusão de h e gamma
Ac = Aa - Ba*K*Ca;
Ac= [Ac, zeros(9,1);
     Vtrim*(Ctheta-Calpha) zeros(1,5)];
 
Bc = Bagamma *(-2.5*pi/180);
Bc = [Bc;0];

Cc = [Ca zeros(7,1)];

Dc = [zeros(7,1)];

sys_fb = ss(Ac,Bc,Cc,Dc);
damp(sys_fb)
[Y,T,X] = step(sys_fb,134);

figure(5)
subplot(3,2,1)
plot(T,X(:,1),'-b','linewidth',1)
grid on
ylabel('u [ft/s]')

subplot(3,2,2)
plot(T,(X(:,2)),'-b','linewidth',1)
grid on
ylabel('w [ft/s]')

subplot(3,2,3)
plot(T,radtodeg(X(:,3)),'-b','linewidth',1)
grid on
ylabel('q [deg/s]')

subplot(3,2,4)
plot(T,radtodeg(X(:,4)),'-b','linewidth',1)
grid on
ylabel('\theta [deg]')

subplot(3,2,5)
plot(T,(X(:,5)),'-b','linewidth',1)
grid on
ylabel('Pow [%]')
xlabel('T [s]')

subplot(3,2,6)
plot(T,radtodeg(X(:,6)),'-b','linewidth',1)
grid on
ylabel('de [deg]')
xlabel('T [s]')
%%
% % fig5 = figure(5)
% % % export to pdf
% % set(fig5,'Units','Inches');
% % pos = get(fig5,'Position');
% % set(fig5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% % print(fig5,'fig5','-dpdf','-r0')
%%
figure(6)
subplot(2,2,1)
plot(T,X(:,7),'-b','linewidth',1)
grid on
ylabel('d [ft]')

subplot(2,2,2)
plot(T,X(:,8),'-b','linewidth',1)
grid on
ylabel('Ev')

subplot(2,2,3)
plot(T,X(:,9),'-b','linewidth',1)
grid on
ylabel('Ed')

subplot(2,2,4)
plot(T,3000+X(:,10),'-b','linewidth',1)
grid on
ylabel('h [ft]')
%%
% % fig6 = figure(6)
% % % export to pdf
% % set(fig6,'Units','Inches');
% % pos = get(fig6,'Position');
% % set(fig6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
% % print(fig6,'fig6','-dpdf','-r0')
%%
% Almaçeando os estados da simulação para empregar como condição inicial
% no flare
Xu_0 = X(end,1);
Xw_0 = X(end,2);
Xq_0 = X(end,3);
Xt_0 = X(end,4);
XP_0 = X(end,5);
Xd_e_0 = X(end,6);
Xd_0 = X(end,7);
XEv_0 = X(end,8);
XEd_0 = X(end,9);
Xh_0 = X(end,10);
%% ==============================FLARE=====================================
%% Montagem de matrizes longitudinais para flare
sel_f =  [1 2 3 4 5];
A_l = [Along Blong_p Alin(sel_long,9); 
     zeros(1,5) -20.2 0;
     Alin(9,sel_long) 0 Alin(9,9)];
 
H = [Hlong 0 0;
    zeros(1,6) 1 -1];

% Matrizes aumentadas
A_l = [A_l zeros(7,1)  zeros(7,2)
    zeros(1,7) -1/2 0 0
    -G*H, F];
rowNames = {'u','w','q','theta','Pow','de','h','href','Ev','Eh'};
colNames = {'u','w','q','theta','Pow','de','h','href','Ev','Eh'};
Al_label = array2table(A_l,'RowNames',rowNames,'VariableNames',colNames)

B_l = [B
    0 0
    0 0
    0 0];

rowNames = {'u','w','q','theta','Pow','de','h','href','Ev','Eh'};
colNames = {'d_t','d_e'};
Bl_label = array2table(B_l,'RowNames',rowNames,'VariableNames',colNames)


C_l = [C]; 
rowNames = {'alpha','theta','q'};
colNames = {'u','w','q','theta','Pow','d_p','h'};
C_label = array2table(C_l,'RowNames',rowNames,'VariableNames',colNames)

G = eye(2);
F = zeros(2);
D = [0,0;1,0;0,1;0 0];
J = [1,0;0,0;0,0;0 1];

C_l = [C_l zeros(3,3);
     -J*H,D];
rowNames = {'alpha','theta','q','ev','Ev','Eh','eh'};
colNames = {'u','w','q','theta','Pow','de','h','href','Ev','Eh'};
C_a_label = array2table(C_l,'RowNames',rowNames,'VariableNames',colNames)

G_l = [0 0
    Ga];
F_l = Fa;
H_l = [H zeros(2,2)];
%% Empregando ganhos obtidos do approach para realimentar flare
k1  = K(2,1);       %alpha
k2  = K(2,2);       %theta
k3  = K(2,3);       %q
k4  = K(1,4);       %ev
k5  = K(1,5);       %intev         
k6  = -0.001;      %inteh
k7  = -0.1;      %eh
%===================================================================
k_ini = [0 0  0  k4  k5  0   0 ;
        k1 k2 k3 0   0   k6  k7];
    
sys_a = ss(A_l-B_l*k_ini*C_l,B_l(:,1),C_l(5,:),0);
% rlocus(sys_a)
damp(sys_a);
%% Otimização de ganhos 
Q = H_l'*H_l;
q =0.01;
R =[1,0;0,10];
V = 0;
r0 = [1;0];

opt = optimset('Display','iter','TolX',1e-12,'TolFun',1e-12);
% Obtendo a matriz de ganhos estabilizantes
K0 = [k_ini(1,4),k_ini(1,5),k_ini(2,1),k_ini(2,2),k_ini(2,3),k_ini(2,6),k_ini(2,7)];
K1=fminsearch(@PI2,K0,opt,A_l,B_l,C_l,Fa,G_l,H_l,Q,R,r0,V,q);

K = [0,0,0,K1(1),K1(2), 0,0;
    K1(3),K1(4),K1(5),0,0 K1(6),K1(7)];

Ac = A_l-B_l*K*C_l;
rowNames = {'u','w','q','theta','Pow','de','h','href','Ev','Eh'};
colNames = {'u','w','q','theta','Pow','de','h','href','Ev','Eh'};
Ac_label = array2table(Ac,'RowNames',rowNames,'VariableNames',colNames)

Cc = C_l;

Bgamma = [zeros(7,1)
    -Vtrim];
Bcgamma =  Bgamma*(-2.5*pi/180);
Bc = [Bcgamma
zeros(2,1)];

sys_fb = ss(Ac,Bc,Cc,0);
damp(sys_fb)
[Y,T,X] = initial(sys_fb,[Xu_0,Xw_0,Xq_0,Xt_0,XP_0,Xd_e_0,3000-(-Xh_0),3000-(-Xh_0),XEv_0,0],50);


figure(8)
subplot(3,2,1)
plot(T,X(:,1),'-b','linewidth',1)
grid on
ylabel('u [ft/s]')

subplot(3,2,2)
plot(T,(X(:,2)),'-b','linewidth',1)
grid on
ylabel('w [ft/s]')

subplot(3,2,3)
plot(T,radtodeg(X(:,3)),'-b','linewidth',1)
grid on
ylabel('q [deg/s]')

subplot(3,2,4)
plot(T,radtodeg(X(:,4)),'-b','linewidth',1)
grid on
ylabel('\theta [deg]')

subplot(3,2,5)
plot(T,(X(:,5)),'-b','linewidth',1)
grid on
ylabel('Pow [%]')
xlabel('T [s]')

subplot(3,2,6)
plot(T,(X(:,6)),'-b','linewidth',1)
grid on
ylabel('de [deg]')
xlabel('T [s]')
%%
fig8 = figure(8);
% export to pdf
set(fig8,'Units','Inches');
pos = get(fig8,'Position');
set(fig8,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig8,'fig8','-dpdf','-r0')
%%
figure(9)
plot(T,X(:,7),'-b',T,X(:,8),'--r','linewidth',1)
grid on
ylabel('h & href [ft]')
xlabel('T [s]')
%%
fig9 = figure(9);
% export to pdf
set(fig9,'Units','Inches');
pos = get(fig9,'Position');
set(fig9,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig9,'fig9','-dpdf','-r0')

