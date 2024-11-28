function [Xdot,Y] = dynamics(t,X,U)

global g
global aircraft
global rad2deg
% X = [u v w p q r x y z phi theta psi | P].';

u = X(1);
v = X(2);
w = X(3);
p_rad_s = X(4);
q_rad_s = X(5);
r_rad_s = X(6);
x = X(7);
y = X(8);
z = X(9);
phi_rad = X(10);
theta_rad = X(11);
psi_rad = X(12); 


omega_b = [p_rad_s q_rad_s r_rad_s].';
V_b = [u v w].';

C_phi = Cmat(1,phi_rad);
C_theta = Cmat(2,theta_rad);
C_psi = Cmat(3,psi_rad);

C_bv = C_phi*C_theta*C_psi;

g_b = C_bv*[0 0 g].';

m = aircraft.m;
J_O_b = aircraft.J_O_b;
rC_b = aircraft.rC_b;

Mgen = [m*eye(3) -m*skew(rC_b)
        m*skew(rC_b) J_O_b];

% Cargas aerodinâmicas e propulsivas:

[Faero_b,Maero_O_b,Yaero] = aero_loads(X,U);
[Fprop_b,Mprop_O_b,Yprop] = prop_loads(X,U);

% Termos restantes das EdMs:

% Primeiro membro (equação das forças):
eq_F = m*skew(omega_b)*V_b +...
    -m*skew(omega_b)*skew(rC_b)*omega_b;

% Segundo membro menos o primeiro:
eq_F = -eq_F +...
    Faero_b + Fprop_b + m*g_b;
                                
% Primeiro membro (equação dos momentos):

eq_M = skew(omega_b)*J_O_b*omega_b +...
    m*skew(rC_b)*skew(omega_b)*V_b;

% Segundo membro menos o primeiro:
eq_M = -eq_M +...
    Maero_O_b + Mprop_O_b + m*skew(rC_b)*g_b;

% Mgen*edot = [eq_F; eq_M];
edot = Mgen\[eq_F; eq_M];

% edot = [udot vdot wdot pdot qdot rdot]

% Cinemática de rotação:
HPhi_inv = [C_phi(:,1) C_phi(:,2) C_bv(:,3)];
Phi_dot_rad_s = HPhi_inv\omega_b; %Taxas angulares em rad/seg.

% Cinemática de translação:

dREOdt = C_bv.'*V_b;


% X = [u v w p q r x y z phi theta psi | P].';


V = Yaero(1);
alpha_deg = Yaero(2);
q_deg_s = q_rad_s*rad2deg;
theta_deg = theta_rad*rad2deg;
H = -z;

beta_deg = Yaero(3);
phi_deg = phi_rad*rad2deg;
p_deg_s = p_rad_s*rad2deg;
r_deg_s = r_rad_s*rad2deg;
psi_deg = psi_rad*rad2deg;

Vdot = (V_b.'*edot(1:3))/V;
udot = edot(1);
vdot = edot(2);
wdot = edot(3);
alpha_dot_rad_s = (u*wdot-w*udot)/(u^2+w^2);
beta_dot_rad_s = (V*vdot-v*Vdot)/(V*sqrt(u^2+w^2));

n_C_b = -1/(m*g)*(Faero_b + Fprop_b);

r_pilot_b = aircraft.r_pilot_b;
n_pilot_b = n_C_b + ...
    -1/g*(skew(edot(4:6))*(r_pilot_b-rC_b)+skew(omega_b)*skew(omega_b)*(r_pilot_b-rC_b));

[rho,~,~,a] = atmosphere(H);

Mach = V/a;

qbar = 0.5*rho*V^2;

gamma = -2.5;

d_dot = 500*(theta_deg-alpha_deg-gamma);

tau_flare = 2;

if t == 70;
   h_dot = 70/tau_flare;
else
h_dot = -(1/tau_flare)*X(20);
end
Xdot = [edot
    dREOdt
    Phi_dot_rad_s
    Yprop(1)
    X(14)
    X(15)
    X(16)
    X(17)
    X(18)
    d_dot
    h_dot
    X(21)
    X(22)
    X(23)
    X(24)]; 

h_dot = V*sind(theta_deg-alpha_deg);

Y = [V
    alpha_deg
    q_deg_s
    theta_deg
    H
    x
    beta_deg
    phi_deg
    p_deg_s
    r_deg_s
    psi_deg
    y
    n_pilot_b
    n_C_b
    Mach
    qbar
    Yprop(2:end)
    Yaero(4:end)
    Vdot
    alpha_dot_rad_s
    beta_dot_rad_s
    d_dot
    h_dot];

end