function X = state_vec(x,trim_par)

% x = [u v w p q r x y z phi theta psi power erro | throttle de da dr]

X = zeros(24,1);

X(1:6) = x(1:6);

X(9) = -trim_par.H_ft;

X(10:12) = x(7:9);

X(13) = x(10);

% erro H
X(14) =0;
% erro d
X(15) =0;
% erro V
X(16) =0;


% delta_e
X(17) = 0;
% delta_t
X(18) = 0;
% d
X(19) = 0;
% h
X(20) = 0;

% laterais
%erro phi
X(21) = 0;
%erro r
X(22) = 0;

% delta_a
X(23) = 0;
% delta_r
X(24) = 0;

