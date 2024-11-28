clear all
close all
clc
global g aircraft trim_par

g                           = 9.80665;
run GNBA.m;

%% -----------------------Cálculo de equilíbrio---------------------------%
g                           = 9.80665;
xCG                         = 0*c;
yCG                         = 0;
zCG                         = 0;
rC_b                        = [xCG yCG zCG].';
r_pilot_b                   = [0; 0; 0];
H_eq                        = distdim(38000,'ft','m');
Mach_eq                     = .78;
[rho,~,~,a]                 = ISA(H_eq);
V_eq                        = a*Mach_eq;
gamma_deg_eq                = 0;
phi_dot_deg_s_eq            = 0;
theta_dot_deg_s_eq          = 0;
psi_dot_deg_s_eq            = V_eq^2/1D4;
trim_par                    = struct('V',V_eq,'H',H_eq,'chi_deg',0,'gamma_deg',gamma_deg_eq,'phi_dot_deg_s',phi_dot_deg_s_eq,'theta_dot_deg_s',theta_dot_deg_s_eq,'psi_dot_deg_s',psi_dot_deg_s_eq); 
x_eq_0                      = zeros(13,1);
x_eq_0(1)                   = V_eq;
x_eq_0(5)                   = H_eq;
options                     = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,'MaxFunEvals',inf,'MaxIter',inf);
[x_eq,fval,exitflag,output,jacobian] = fsolve(@trimGNBA,x_eq_0,options,trim_par);

X_eq                        = state_vec(x_eq,trim_par);
control_vec                 = @(x)(x(10:13)); 
control                     = control_vec(x_eq);
U_eq                        = zeros(6,1);
U_eq(1:2)                   = control(1)/2;
U_eq(3)                     = control(2);
U_eq(4)                     = 0;
U_eq(5)                     = control(3);
U_eq(6)                     = control(4);
[Xdot_eq,Y_eq]              = dynamics(0,X_eq,U_eq);

fprintf('----- TRIMMED FLIGHT PARAMETERS -----\n\n');
fprintf('   %-10s = %10.4f %-4s\n','x_CG',xCG,'m');
fprintf('   %-10s = %10.4f %-4s\n','y_CG',yCG,'m');
fprintf('   %-10s = %10.4f %-4s\n','z_CG',zCG,'m');
fprintf('   %-10s = %10.4f %-4s\n','gamma',trim_par.gamma_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','chi',trim_par.chi_deg,'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi_dot',trim_par.phi_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta_dot',trim_par.theta_dot_deg_s,'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi_dot',trim_par.psi_dot_deg_s,'deg/s');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','V',X_eq(1),'m/s');
fprintf('   %-10s = %10.4f %-4s\n','alpha',X_eq(2),'deg');
fprintf('   %-10s = %10.4f %-4s\n','q',X_eq(3),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','theta',X_eq(4),'deg');
fprintf('   %-10s = %10.1f %-4s\n','H',X_eq(5),'m');
fprintf('   %-10s = %10.4f %-4s\n','beta',X_eq(7),'deg');
fprintf('   %-10s = %10.4f %-4s\n','phi',X_eq(8),'deg');
fprintf('   %-10s = %10.4f %-4s\n','p',X_eq(9),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','r',X_eq(10),'deg/s');
fprintf('   %-10s = %10.4f %-4s\n','psi',X_eq(11),'deg');
fprintf('\n');
fprintf('   %-10s = %10.2f %-4s\n','T_{le}',U_eq(1),'N');
fprintf('   %-10s = %10.2f %-4s\n','T_{re}',U_eq(2),'N');
fprintf('   %-10s = %10.2f %-4s\n','i_h',U_eq(3),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_e',U_eq(4),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_a',U_eq(5),'deg');
fprintf('   %-10s = %10.4f %-4s\n','delta_r',U_eq(6),'deg');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','n_x_pilot',Y_eq(1),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_pilot',Y_eq(2),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_pilot',Y_eq(3),'');
fprintf('   %-10s = %10.4f %-4s\n','n_x_CG',Y_eq(4),'');
fprintf('   %-10s = %10.4f %-4s\n','n_y_CG',Y_eq(5),'');
fprintf('   %-10s = %10.4f %-4s\n','n_z_CG',Y_eq(6),'');
fprintf('\n');
fprintf('   %-10s = %10.4f %-4s\n','Mach',Y_eq(7),'');
fprintf('   %-10s = %10.2f %-4s\n','Dyn. p.',Y_eq(8),'Kg/m^2');

%% -------------------------------Simulação-------------------------------%
tf = 50;
dt = 1e-2;

X0 = X_eq;
tic
        Tsol = 0:dt:tf;
        Xsol = ode4(@dynamics,0:dt:tf,X0,U_eq);
        i_t = 1;
        Usol = zeros(size(Xsol,1),length(U_eq));
        Ysol = zeros(size(Xsol,1),length(Y_eq));
        for u_t=0:dt:tf
        Usol(i_t,:) = U_eq.';
        [~,Y] = dynamics(u_t,Xsol(i_t,:).',U_eq);
        Ysol(i_t,:) = Y.';
        i_t = i_t+1;    
        end
        time_ode4 = toc
%% ---------------------------------Plot----------------------------------%
%X                        = [V alpha q theta H x beta phi p r psi y].';

figure
subplot(3,2,1)
plot(Tsol,Xsol(:,1))
ylabel('V (m/s)')
subplot(3,2,2)
plot(Tsol,Xsol(:,2))
ylabel('\alpha (deg)')
subplot(3,2,3)
plot(Tsol,Xsol(:,3))
ylabel('q (deg/s)')
subplot(3,2,4)
plot(Tsol,Xsol(:,4))
ylabel('\theta (deg)')
subplot(3,2,5)
plot(Tsol,Xsol(:,5))
ylabel('H (m)')
subplot(3,2,6)
plot(Tsol,Xsol(:,6))
ylabel('x (m)')

figure
subplot(3,2,1)
plot(Tsol,Xsol(:,7))
ylabel('\beta (deg)')
subplot(3,2,2)
plot(Tsol,Xsol(:,8))
ylabel('phi (deg)')
subplot(3,2,3)
plot(Tsol,Xsol(:,9))
ylabel('p (deg/s)')
subplot(3,2,4)
plot(Tsol,Xsol(:,10))
ylabel('r (deg/s)')
subplot(3,2,5)
plot(Tsol,Xsol(:,11))
ylabel('psi (deg)')
subplot(3,2,6)
plot(Tsol,Xsol(:,12))
ylabel('y (m)')

figure
subplot(3,2,1)
plot(Tsol,Usol(:,1))
ylabel('Tle (N)')
subplot(3,2,2)
plot(Tsol,Usol(:,2))
ylabel('Tre (N)')
subplot(3,2,3)
plot(Tsol,Usol(:,3))
ylabel('i_h (deg)')
subplot(3,2,4)
plot(Tsol,Usol(:,4))
ylabel('\delta_e (deg)')
subplot(3,2,5)
plot(Tsol,Usol(:,5))
ylabel('\delta_a (deg)')
subplot(3,2,6)
plot(Tsol,Usol(:,6))
ylabel('\delta_e (deg)')

figure
plot3(Xsol(:,12),Xsol(:,6),Xsol(:,5))
xlabel('y [m]')
ylabel('x [m]')
zlabel('H [m]')
grid on

