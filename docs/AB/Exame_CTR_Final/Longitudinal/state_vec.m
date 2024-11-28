function X = state_vec(x,trim_par)

% x = [u v w p q r phi theta psi power | throttle de da dr]

X = zeros(13,1);

X(1:6) = x(1:6);

X(9) = -trim_par.H_ft;

X(10:12) = x(7:9);

X(13) = x(10);