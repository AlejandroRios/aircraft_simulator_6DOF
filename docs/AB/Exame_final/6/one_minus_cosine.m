function [Xdot,Y,U,W] = one_minus_cosine(t,X,U_eq,W_eq)

U = U_eq;
% t_i = 1;
Vg = -5;
x0 = 1000;
L = 2000;

x = X(6);
dW = zeros(3,1);

if x >= x0 && x<=x0+L 
    dW(3) =Vg*(1-cos(2*pi*(x-x0)/L));
end

W = W_eq+dW;

[Xdot,Y] = dynamics(t,X,U,W);
end