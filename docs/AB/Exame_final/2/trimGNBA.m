function f = trimGNBA(x, trim_par)

% dimensão de x è 14 incognitas 
% X = [1   2   3    4   5 6  7    8   9  10  11   12 | 13  14  15 16 17 18]
% X = [V alpha q  theta H x beta phi  p  r   psi  y  | Tle Tre ih de da dr]
% x = [1   2   3    4   5 -  6    7   8  9  -    -   | 10  11  12 13 14 15]
% trim_par parametros de trimagem
X = state_vec(x,trim_par);
U = control_vec(x);

[Xdot,Y] = dynamics(0,X,U,trim_par.W);
global g
Mach = 0.78;
H_m = trim_par.H_m;
[~,~,~,a] = ISA(H_m);
V_eq = Mach*a;
Raio = 10000;
phi_des = atand((V_eq^2)/(g*Raio));
psi_dot_deg_des =  (sind(phi_des)/(cosd(X(7))*(sind(X(2))*tand(X(4))+...
cosd(X(2))*cosd(phi_des)))) * (g/V_eq)

% Velocidade que està sendo utilizada é a velocidade inercial
C_tv = Cmat(2,trim_par.gamma_deg*pi/180)*Cmat(3,trim_par.chi_deg*pi/180);
V_i = C_tv.'*[trim_par.V;0;0];
Beta = X(7);

f = [Xdot(1)
     Xdot(2)
     Xdot(3)
     Xdot(4)-trim_par.theta_dot_deg_s
     Xdot(5) - V_i(3)
     Xdot(6) - V_i(1)
     Xdot(7)
     Xdot(8)-trim_par.phi_dot_deg_s
     Xdot(9)
     Xdot(10)
%      psi_dot_deg_des
     Xdot(12) - V_i(2)
     Beta
     U(1) - U(2)];