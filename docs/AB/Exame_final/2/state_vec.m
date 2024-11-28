function X = state_vec(x,trim_par)

% X = [1   2   3    4   5 6  7    8   9  10  11   12 | 13  14  15 16 17 18]
% X = [V alpha q  theta H x beta phi  p  r   psi  y  | Tle Tre ih de da dr]
% x = [1   2   3    4   5 -  6    7   8  9  -    -   | 10  11  12 13 14 15]
global g
V_eq = 230.15 ;
Raio = 10000;
phi_des = atand((V_eq^2)/(g*Raio));


X = zeros(12,1);

X(1) = x(1); % V
X(2) = x(2); % alpha
X(3) = x(3); % q
X(4) = x(4); % theta
X(5) = trim_par.H_m; % H

X(7) = x(5); % beta
X(8) = phi_des; % phi
X(9) = x(6); % p
X(10) = x(7); % r
X(11) = x(8); % r

