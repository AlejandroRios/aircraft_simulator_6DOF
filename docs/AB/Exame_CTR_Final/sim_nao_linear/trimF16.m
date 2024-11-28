function f = trimF16(x, trim_par)

% dimensão de x è 14 incognitas 
% x = [u v w p q r phi theta psi power | throttle de da dr]
% trim_par parametros de trimagem
erro_passado = 0;
X = state_vec(x,trim_par);

% control_vec = @(x)(x(11:14));
U = control_vec(x);

[Xdot,Y] = dynamics(0,X,U);

C_tv = Cmat(2,trim_par.gamma_deg*pi/180)*Cmat(3,trim_par.chi_deg*pi/180);
V_i = C_tv.'*[trim_par.V;0;0];

n_y_pilot = Y(14);

f = [Xdot(1:6)
    Xdot(7:9)-V_i
    Xdot(10)-trim_par.phi_dot_deg_s*pi/180
    Xdot(11)-trim_par.theta_dot_deg_s*pi/180
    Xdot(12)-trim_par.psi_dot_deg_s*pi/180
    Xdot(13)
    n_y_pilot];