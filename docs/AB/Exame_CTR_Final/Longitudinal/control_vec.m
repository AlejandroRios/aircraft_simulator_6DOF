function U = control_vec(x)

% x = [u v w p q r phi theta psi power | throttle de da dr]

U = x(11:14);
