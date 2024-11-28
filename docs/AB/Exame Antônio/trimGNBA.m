function f=trimGNBA(x,trim_par)
%x = [V alpha q theta H beta phi p r |T ih da dr]';   %Incógnitas
%X = [V alpha q theta H x beta phi p r psi y]
X                       = state_vec(x,trim_par);      %Matriz de estados
control_vec             = @(x)(x(10:13));             %Matriz de Controle
control                 = control_vec(x);
U                       = zeros(6,1);
U(1:2)                  = control(1)/2;
U(3)                    = control(2);
U(4)                    = 0;
U(5)                    = control(3);
U(6)                    = control(4);
[Xdot,Y]                = dynamics(0,X,U);           %Dinâmica da aeronave
beta_deg                = X(7);
f                       = [Xdot(1)
                           Xdot(2)
                           Xdot(3)
                           Xdot(4)-trim_par.theta_dot_deg_s
                           Xdot(5)-0
                           Xdot(6)-trim_par.V
                           Xdot(8)-trim_par.phi_dot_deg_s
                           Xdot(9)
                           Xdot(10)
                           Xdot(11)-trim_par.psi_dot_deg_s
                           Xdot(12)-0
                           beta_deg]; 

                           
end
    