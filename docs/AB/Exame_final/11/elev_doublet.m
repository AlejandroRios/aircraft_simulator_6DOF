function [Xdot,Y,U] = elev_doublet(t,X,U_eq)

U = U_eq;

t_i = 1;
dt_doublet = 4;
ampl = 2;

dU = 0;
if t > t_i && t<=t_i+dt_doublet/2
    dU = ampl;
elseif t>t_i+dt_doublet/2 && t<=t_i+dt_doublet
    dU = -ampl;
end

% no elevador
U(4) = U(4)+dU;  

%no leme
% U(4) = U(4)+dU;  

[Xdot,Y] = dynamics(t,X,U);
end