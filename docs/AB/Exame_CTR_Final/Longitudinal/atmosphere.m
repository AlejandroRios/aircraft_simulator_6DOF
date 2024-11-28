function [rho,T,p,a] = atmosphere(alt)

R0 = 2.377e-3;

Tfac = 1.0-0.703e-5*alt;
T = 519.0*Tfac;

if alt>=35000, T = 390.0; end

rho = R0*(Tfac^4.14);
a = sqrt(1.4*1716.3*T);
p = 1715.0*rho*T;