function [Xdot,Y,U,W] = one_minus_cosine(t,X,U_eq,W_eq)

U = U_eq;
% t_i = 1;
Vg = -30;
x0 = 5000;
L = 15000;

x = X(7);
dW = zeros(3,1);

if x >= x0 && x<=x0+L 
    dW(3) =Vg*(1-cos(2*pi*(x-x0)/L));
end

W = W_eq+dW;

[Xdot,Y] = dynamics(t,X,U,W);
end