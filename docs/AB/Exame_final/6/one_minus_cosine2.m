function [Xdot,Y,U,Wind] = one_minus_cosine2(t,X,U_eq,W_eq)

U = U_eq;
% t_i = 1;
Vg = -5;
x0 = 1000;
L = 2000;

x = X(6);

V = X(1);
alpha_deg = X(2);
beta_deg = X(7);
theta_deg = X(4);
theta_rad = degtorad(theta_deg);
phi_deg = X(8);
phi_rad = degtorad(phi_deg);
psi_deg = X(11);
psi_rad = degtorad(psi_deg);

urel = V*cosd(beta_deg)*cosd(alpha_deg);
vrel = V*sind(beta_deg);
wrel = V*cosd(beta_deg)*sind(alpha_deg);


C_phi = Cmat(1,phi_rad);
C_theta = Cmat(2,theta_rad);
C_psi = Cmat(3,psi_rad);

C_bv = C_phi*C_theta*C_psi;


dW = zeros(6,1);
if x >= x0 && x<=x0+L 
    dW(3) =Vg*(1-cos(2*pi*(x-x0)/L));  
    Vw_b = C_bv*(W_eq(1:3)+dW(1:3));
    Vrel_b = [urel vrel wrel].';
    V_b = Vrel_b + Vw_b;
    
    dREOdt = C_bv.'*V_b;
    dxdt = dREOdt(1);
    dW(6) = Vg*(sin(2*pi*(x-x0)/L))*2*pi/L*dxdt;
end

Wind = W_eq+dW;

[Xdot,Y] = dynamics(t,X,U,Wind);


