
fprintf('----- TRIMMED FLIGHT PARAMETERS -----\n\n');
fprintf('   %-10s = %10.4f %-4s\n','x_CG',xCG,'ft');
fprintf('   %-10s = %10.4f %-4s\n','y_CG',yCG,'ft');
fprintf('   %-10s = %10.4f %-4s\n','z_CG',zCG,'ft');
fprintf('   %-10s = %10.4f %-4s\n','gamma',trim_par.gamma_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','chi',trim_par.chi_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi_dot',trim_par.phi_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta_dot',trim_par.theta_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi_dot',trim_par.psi_dot_deg_s,'deg/s');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','V',Y_eq(1),'ft/s');
fprintf('   %-10s = %10.4f %-4s\n','alpha',Y_eq(2),'deg');
fprintf('   %-10s = %10.4f %-4s\n','q',Y_eq(3),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta',Y_eq(4),'deg');
fprintf('   %-10s = %10.1f %-4s\n','H',Y_eq(5),'ft');
fprintf('   %-10s = %10.4f %-4s\n','beta',Y_eq(7),'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi',Y_eq(8),'deg');
fprintf('   %-10s = %10.4f %-4s\n','p',Y_eq(9),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','r',Y_eq(10),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi',Y_eq(11),'deg');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','throttle',100*U_eq(1),'%');
fprintf('   %-10s = %10.4f %-4s\n','delta_e',U_eq(2),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_a',U_eq(3),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_r',U_eq(4),'deg');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','n_x_pilot',Y_eq(13),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_pilot',Y_eq(14),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_pilot',Y_eq(15),'');
fprintf('   %-10s = %10.4f %-4s\n','n_x_CG',Y_eq(16),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_CG',Y_eq(17),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_CG',Y_eq(18),'');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','Mach',Y_eq(19),'');
fprintf('   %-10s = %10.2f %-4s\n','Dyn. p.',Y_eq(20),'lbf/ft^2');





tic
options = odeset('MaxStep',dt);
% [Tsol,Xsol] = ode45(@dynamics,0:dt:tf,X_eq,[],U_eq);
[Tsol,Xsol] = ode45(@dynamics,0:dt:tf,X_eq,options,U_eq);
i_t = 1;
Usol = zeros(size(Xsol,1),length(U_eq));
Ysol = zeros(size(Xsol,1),length(Y_eq));
for u_t=0:dt:tf
    Usol(i_t,:) = U_eq.';
    [~,Y] = dynamics(u_t,Xsol(i_t,:).',U_eq);
    Ysol(i_t,:) = Y.';
    i_t = i_t+1;
end
time_ode45 = toc

tic
Tsol_ode4 = 0:dt:tf;
Xsol_ode4 = ode4(@dynamics,0:dt:tf,X_eq,U_eq);
i_t = 1;
Usol_ode4 = zeros(size(Xsol_ode4,1),length(U_eq));
Ysol_ode4 = zeros(size(Xsol_ode4,1),length(Y_eq));
for u_t=0:dt:tf
    Usol_ode4(i_t,:) = U_eq.';
    [~,Y] = dynamics(u_t,Xsol_ode4(i_t,:).',U_eq);
    Ysol_ode4(i_t,:) = Y.';
    i_t = i_t+1;
end
time_ode4 = toc





V = Yaero(1);
alpha_deg = Yaero(2);
q_deg_s = q_rad_s*rad2deg;
theta_deg = theta_rad*rad2deg;
H = -z;

beta_deg = Yaero(3);
phi_deg = phi_rad*rad2deg;
p_deg_s = p_rad_s*rad2deg;
r_deg_s = r_rad_s*rad2deg;
psi_deg = psi_rad*rad2deg;

Vdot = (V_b.'*edot(1:3))/V;
udot = edot(1);
vdot = edot(2);
wdot = edot(3);
alpha_dot_rad_s = (u*wdot-w*udot)/(u^2+w^2);
beta_dot_rad_s = (V*vdot-v*Vdot)/(V*sqrt(u^2+w^2));

n_C_b = -1/(m*g)*(Faero_b + Fprop_b);

r_pilot_b = aircraft.r_pilot_b;
n_pilot_b = n_C_b + ...
    -1/g*(skew(edot(4:6))*(r_pilot_b-rC_b)+skew(omega_b)*skew(omega_b)*(r_pilot_b-rC_b));

[rho,~,~,a] = atmosphere(H);

Mach = V/a;

qbar = 0.5*rho*V^2;

Y = [V
    alpha_deg
    q_deg_s
    theta_deg
    H
    x
    beta_deg
    phi_deg
    p_deg_s
    r_deg_s
    psi_deg
    y
    n_pilot_b
    n_C_b
    Mach
    qbar
    Yprop(2:end)
    Yaero(4:end)
    Vdot
    alpha_dot_rad_s
    beta_dot_rad_s];




