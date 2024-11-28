function f = trimGNBA(x, trim_par)

% dimensão de x è 14 incognitas 
% X = [1   2   3    4   5 6  7    8   9  10  11   12 | 13  14  15 16 17 18]
% X = [V alpha q  theta H x beta phi  p  r   psi  y  | Tle Tre ih de da dr]
% x = [1   2   3    4   5 -  6    7   8  9  -    -   | 10  11  12 13 14 15]
% trim_par parametros de trimagem
X = state_vec(x,trim_par);
U = control_vec(x);

[Xdot,Y] = dynamics(0,X,U);

% Velocidade que està sendo utilizada é a velocidade inercial
C_tv = Cmat(2,trim_par.gamma_deg*pi/180)*Cmat(3,trim_par.chi_deg*pi/180);
V_i = C_tv.'*[trim_par.V;0;0];


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
     Xdot(11)-trim_par.psi_dot_deg_s
     Xdot(12) - V_i(2)
     Y(7)
     U(1) - U(2)];