function [Xdot,Y] = dynamics(t,X,U,W)

global g
global aircraft
global rad2deg
% x = [||u v w p q r || x y ||z|| ||phi theta psi|| power | throttle de da dr]
% x = [V alpha q  theta H x beta phi p r psi y | Tle Tre ih de da dr]

V = X(1);
alpha_deg = X(2);

q_deg_s = X(3);
q_rad_s = degtorad(q_deg_s);

theta_deg = X(4);
theta_rad = degtorad(theta_deg);

H_m = X(5);
x = X(6);
beta_deg = X(7);

phi_deg = X(8);
phi_rad = degtorad(phi_deg);

p_deg_s = X(9);
p_rad_s = degtorad(p_deg_s);

r_deg_s = X(10);
r_rad_s = degtorad(r_deg_s);

psi_deg = X(11);
psi_rad = degtorad(psi_deg);

y = X(12); 





urel = V*cosd(beta_deg)*cosd(alpha_deg);
vrel = V*sind(beta_deg);
wrel = V*cosd(beta_deg)*sind(alpha_deg);

V = sqrt(urel^2 + vrel^2 + wrel^2);

alpha_rad = atan(wrel/urel);
alpha_deg = alpha_rad*180/pi;

beta_rad = asin(vrel/V);
beta_deg = beta_rad*180/pi;


omega_b = [p_rad_s q_rad_s r_rad_s].';
V_b = [urel vrel wrel].';

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
% para vento rel mexemos nesta chamada
% Xaero = [urel vrel wrel p q r x y z phi theta psi | P].';
Xaero = X;
Vw_b = C_bv*W(1:3);
dVwdt_b = C_bv*W(4:6);
Vrel_b = [urel vrel wrel].';
V_b = Vrel_b + Vw_b;
% 
% [Faero_b,Maero_O_b,Yaero] = aero_loads(Xaero,U,Vrel_b);
% [Fprop_b,Mprop_O_b,Yprop] = prop_loads(Xaero,U,Vrel_b);

[Faero_b,Maero_O_b,Yaero] = aero_loads(X,U);
[Fprop_b,Mprop_O_b,Yprop] = prop_loads(X,U);

% Termos restantes das EdMs:

% Primeiro membro (equação das forças):
eq_F = m*skew(omega_b)*Vrel_b +...
    -m*skew(omega_b)*skew(rC_b)*omega_b;

% Segundo membro menos o primeiro:
eq_F = -eq_F +...
    Faero_b + Fprop_b + m*g_b + ...
    -m*dVwdt_b;
                                
% Primeiro membro (equação dos momentos):

eq_M = skew(omega_b)*J_O_b*omega_b +...
    m*skew(rC_b)*skew(omega_b)*Vrel_b;

% Segundo membro menos o primeiro:
eq_M = -eq_M +...
    Maero_O_b + Mprop_O_b + m*skew(rC_b)*g_b + ...
    -m*skew(rC_b)*dVwdt_b;

% Mgen*edot = [eq_F; eq_M];
edot = Mgen\[eq_F; eq_M];
u_dot = edot(1);
v_dot = edot(2);
w_dot = edot(3);

% edot = [udot vdot wdot pdot qdot rdot]

% Cinemática de rotação:
HPhi_inv = [C_phi(:,1) C_phi(:,2) C_bv(:,3)];
Phi_dot_rad_s = HPhi_inv\omega_b; %Taxas angulares em rad/seg.

% Cinemática de translação:

dREOdt = C_bv.'*V_b;



Mach = 0.78;
[~,~,~,a] = ISA(H_m);
V_eq = Mach*a;
Raio = 10000;
phi_des = atand((V_eq^2)/(g*Raio));

% V_dot = 0;
V_dot = (V_b.'*edot(1:3))/V;
alfa_dot_deg_s = rad2deg*((urel*w_dot-wrel*u_dot)/(urel^2+wrel^2));
q_dot_deg = rad2deg*edot(5);
theta_dot_deg = rad2deg*Phi_dot_rad_s(2);
H_dot = dREOdt(3);
x_dot = dREOdt(1);
beta_dot_deg_s = rad2deg*((V*v_dot-vrel*V_dot)/(V*sqrt(urel^2+wrel^2)));
phi_dot_deg = rad2deg*Phi_dot_rad_s(1);
p_dot_deg_s = rad2deg*edot(4);
r_dot_deg_s = rad2deg*edot(6);
psi_dot_deg = rad2deg*Phi_dot_rad_s(3);
% psi_dot_deg = psi_dot_deg_des;
y_dot_deg = dREOdt(2);

% psi_dot_deg = (g/V)*tand(X(8));

% X = [u v w p q r x y z phi theta psi | P].';
Xdot = [V_dot
    alfa_dot_deg_s
    q_dot_deg 
    theta_dot_deg
    H_dot 
    x_dot
    beta_dot_deg_s
    phi_dot_deg 
    p_dot_deg_s 
    r_dot_deg_s 
    psi_dot_deg
    y_dot_deg]; 




n_C_b = -1/(m*g)*(Faero_b + Fprop_b);

r_pilot_b = aircraft.r_pilot_b;
n_pilot_b = n_C_b + ...
    -1/g*(skew(edot(4:6))*(r_pilot_b-rC_b)+skew(omega_b)*skew(omega_b)*(r_pilot_b-rC_b));

[rho,~,~,a] = ISA(H_m);

Mach = V/a;

qbar = 0.5*rho*V^2;

Y = [V
    alpha_deg
    q_deg_s
    theta_deg
    H_m
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
    Fprop_b
    Mprop_O_b
    Yaero
    V_dot
    alfa_dot_deg_s
    beta_dot_deg_s
    V_b ];

