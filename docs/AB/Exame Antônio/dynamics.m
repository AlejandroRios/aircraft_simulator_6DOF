function [Xdot,Y]=dynamics(t,X,U)
global g 
global aircraft
%% -------------------------------Estados---------------------------------%
%X                        = [V alpha q theta H x beta phi p r psi y].';
V                         = X(1);
alpha_deg                 = X(2);
q_deg_s                   = X(3);
theta_deg                 = X(4);
H                         = X(5);
x                         = X(6);
beta_deg                  = X(7);
phi_deg                   = X(8);
p_deg_s                   = X(9);
r_deg_s                   = X(10);
psi_deg                   = X(11);
y                         = X(12);

%% -----------------------------------------------------------------------%
omega_b                   = [p_deg_s q_deg_s r_deg_s].';  %Velocidade ângular
v                         = V*sind(beta_deg);
u                         = V*cosd(beta_deg)*cosd(alpha_deg);
w                         = V*cosd(beta_deg)*sind(alpha_deg);
V_b                       = [u v w].';                    %Velocidade linear

%% ----------------------Matriz de Transformação--------------------------%
C_phi                     = Cmat(1,degtorad(phi_deg));
C_theta                   = Cmat(2,degtorad(theta_deg));
C_psi                     = Cmat(3,degtorad(psi_deg));
C_bv                      = C_phi*C_theta*C_psi;

%% ------------------------Matriz de Gravidade----------------------------%
g_b                       = C_bv*[0 0 g].';

%% --------------------Matriz de Massa Generalizada-----------------------%
m                         = aircraft.m;                   %Massa da Aeronave
J_O_b                     = aircraft.J_O_b;               %Matriz de Inércia
rC_b                      = aircraft.rC_b;                %Distância entre origem e centro de massa
Mgen                      = [m*eye(3) -m*skew(rC_b)
                             m*skew(rC_b) J_O_b];

%% -------------------------Forças e Momentos-----------------------------%
[Faero_b,Maero_O_b,Yaero] = aero_loads(X,U);
[Fprop_b,Mprop_O_b,Yprop] = prop_loads(X,U);

%% -----------Termos restantes das equações do movimento------------------%
eq_F                      = -(m*skew(omega_b)*V_b - m*skew(omega_b)*skew(rC_b)*omega_b) + Faero_b + Fprop_b + m*g_b;
eq_M                      = -(skew(omega_b)*J_O_b*omega_b + m*skew(rC_b)*skew(omega_b)*V_b) + Maero_O_b + Mprop_O_b + m*skew(rC_b)*g_b;

%% ----------------------------Acelerações--------------------------------%
edot                      = Mgen\[eq_F; eq_M];
u_dot                     = edot(1);
v_dot                     = edot(2);
w_dot                     = edot(3);
Vdot                      = (V_b.'*edot(1:3))/V;

%% ----------------------Cinemática de Rotação----------------------------%
HPhi_inv                  = [C_phi(:,1) C_phi(:,2) C_bv(:,3)];
Phi_dot_rad               = HPhi_inv\omega_b;

%% ---------------------Cinemática de Translação--------------------------%
dReodt                    = C_bv.'*V_b;

%% -----------------------------Fator de Carga----------------------------%
n_C_b                     = -1/(m*g)*(Faero_b + Fprop_b);
r_pilot_b                 = aircraft.r_pilot_b;
n_pilot_b                 = n_C_b + -1/g*(skew(edot(4:6))*(r_pilot_b-rC_b)+skew(omega_b)*skew(omega_b)*(r_pilot_b-rC_b));

%% ---------------------------Pressão Dinâmica----------------------------%
[rho,~,~,a]               = ISA(H);
Mach                      = V/a;
q_bar                     = 0.5*rho*V^2;

%% ------------------------------Saída------------------------------------%
p_deg_dot                 = radtodeg(edot(4));
q_deg_dot                 = radtodeg(edot(5));
r_deg_dot                 = radtodeg(edot(6));
alpha_dot                 = radtodeg((u*w_dot-w*u_dot)/(u^2+w^2));
beta_dot                  = radtodeg((V*v_dot-v*Vdot)/(V*sqrt(u^2+w^2)));
phi_dot                   = radtodeg(Phi_dot_rad(1));
theta_dot                 = radtodeg(Phi_dot_rad(2));
psi_dot                   = radtodeg(Phi_dot_rad(3));
H_dot                     = -dReodt(3); 
V_dot                     = Vdot;
x_dot                     = dReodt(1);
y_dot                     = dReodt(2);
Xdot                      = [V_dot alpha_dot q_deg_dot theta_dot H_dot x_dot beta_dot phi_dot p_deg_dot r_deg_dot psi_dot y_dot].';
Y                         = [n_pilot_b; n_C_b; Mach; q_bar; Yaero; Yprop;beta_deg];
end