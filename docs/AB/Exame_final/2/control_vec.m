function U = control_vec(x)

% X = [1   2   3    4   5 6  7    8   9  10  11   12 | 13  14  15 16 17 18]
% X = [V alpha q  theta H x beta phi  p  r   psi  y  | Tle Tre ih de da dr]
% x = [1   2   3    4   5 -  6    7   8  9  -    -   | 10  11  12 13 14 15]
U = zeros(6,1);
U(1) = x(9);
U(2) = x(10);
U(3) = x(11);
U(5) = x(12);
U(6) = x(13);
