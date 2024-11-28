function [Xdot,Y,U,W] = sharp_edged_gust(t,X,U_eq,W_eq)

U = U_eq;
t_i = 1;
dW = zeros(3,1);
if t >= t_i 
%     dW(3) = -30;
    dW(3) = 0;
end

W = W_eq+dW;

[Xdot,Y] = dynamics(t,X,U,W);
end