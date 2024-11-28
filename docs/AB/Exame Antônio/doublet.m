function [Xdot,Y,U]=doublet(t,X,U_eq,superficie,ampl,dt_doublet,t_i)

U = U_eq;

dt_trans = 1;


t0 = t_i;
t1 = t_i+dt_trans;
t2 = t_i+dt_trans+1/2*(dt_doublet-4*dt_trans);
t3 = t_i+dt_trans+1/2*(dt_doublet-4*dt_trans)+2*dt_trans;
t4 = t_i+dt_trans+1/2*(dt_doublet-4*dt_trans)+2*dt_trans+1/2*(dt_doublet-4*dt_trans);
t5 = t_i+dt_trans+1/2*(dt_doublet-4*dt_trans)+2*dt_trans+1/2*(dt_doublet-4*dt_trans)+dt_trans;

dU = 0;
if t>t0 && t<=t1
    dU = cubic_transition(t,t0,t1,0,0,ampl,0);
elseif t>t1 && t<=t2
    dU = ampl;
elseif t>t2 && t<=t3
    dU = cubic_transition(t,t2,t3,ampl,0,-ampl,0);
elseif t>t3 && t<=t4
    dU = -ampl;
elseif t>t4 && t<=t5
    dU = cubic_transition(t,t4,t5,-ampl,0,0,0);
end 

U(superficie) = U(superficie)+dU;

[Xdot,Y] = dynamics(t,X,U);