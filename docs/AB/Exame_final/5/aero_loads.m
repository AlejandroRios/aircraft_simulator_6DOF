function [Faero_b,Maero_O_b,Yaero] = aero_loads(X,U,Vrel)

global aircraft

u = Vrel(1);
v = Vrel(2);
w = Vrel(3);


V = sqrt(u^2 + v^2 + w^2);
alpha_rad = atan(w/u);
alpha_deg = alpha_rad*180/pi;
beta_rad = asin(v/V);
beta_deg = beta_rad*180/pi;


p_deg_s = X(9);
p_rad_s = degtorad(p_deg_s);
q_deg_s = X(3);
q_rad_s = degtorad(q_deg_s);
r_deg_s = X(10);
r_rad_s = degtorad(r_deg_s);

ih_deg = U(3);
de_deg = U(4);
da_deg = U(5);
dr_deg = U(6);


b = aircraft.b;
c = aircraft.c;
S = aircraft.S;
%========================================forças==================================================
%% Coeficiente de sustentação
CL0 = 0.308;
CLalpha = 0.133;
CLq = 16.7;
CLih = 0.0194;
CLde = 0.00895;
CLalpha_dot = 8.87;
CLq_dot = -9.34;

CL = CL0 + CLalpha*alpha_deg + CLq*((q_rad_s*c)/(2*V)) + CLih*ih_deg + CLde*de_deg;
% CL = CL0 + CLalpha*alpha_deg + CLq*((q_rad_s*c)/(2*V)) + CLih*ih_deg + CLde*de_deg + CLalpha_dot*((alpha_dot*c)/(2*V)) + CLq_dot*((q_dot*c^2)/(4*V^2)) ;
%% Coeficiente de arrastre
CD0 = 0.02207;
CDalpha = 0.00271;
CDalpha2 = 0.000603;
CDq2 = 35.904;
CDbeta2 = 0.000160;
CDp2 = 0.5167;
CDr2 = 0.5738;
CDih = -0.000420;
CDih2 = 0.000134;
CDde2 = 4.61e-5;
CDda2 = 3e-5;
CDdr2 = 1.81e-5;

CD = CD0 + CDalpha*alpha_deg + CDalpha2*alpha_deg^2 + CDq2*((q_rad_s*c)/(2*V))^2 +...
    CDbeta2*beta_deg^2 + CDp2*((p_rad_s*b)/(2*V))^2 + CDr2*((r_rad_s*b)/(2*V))^2 +...
    CDih*ih_deg + CDih2*ih_deg^2 + CDde2*de_deg^2 +...
    CDda2*da_deg^2 + CDdr2*dr_deg^2;
%% Força lateral
CYbeta = 0.0228;
CYp = 0.084;
CYr = -1.21;
CYda = 2.36e-4;
CYdr = -5.75e-3;
CYbeta_dot = 1.64;
CYp_dot = 0.0159;
CYr_dot = -0.0350;


CY = CYbeta*beta_deg + CYp*((p_rad_s*b)/(2*V)) + CYr*((r_rad_s*b)/(2*V)) + ...
    CYda*da_deg + CYdr*dr_deg;
% CY = CYbeta*beta_deg + CYp*((p_rad_s*b)/(2*V)) + CYr*((r_rad_s*b)/(2*V)) + ...
%     CYda*da_deg + CYdr*dr_deg+...
%     CYbeta_dot*((beta_dot*b)/(2*V)) + CYp_dot*((p_dot*b^2)/(4*V^2)) + CYr_dot*((r_dot*b^2)/(4*V^2)) ;
% ===============================Momentos=======================================
%% Coeficiente de momentos de arfagem
Cm0 = 0.0170;
Cmalpha = -0.0402;
Cmq = -57.0;
Cmih = -0.0935;
Cmde = -0.0448;
Cmalpha_dot = -45.8;
Cmq_dot = -14.2;

Cm = Cm0 + Cmalpha*alpha_deg + Cmq*((q_rad_s*c)/(2*V)) + Cmih*ih_deg + Cmde*de_deg;
% Cm = Cm0 + Cmalpha*alpha_deg + Cmq*((q_rad_s*c)/(2*V)) + Cmih*ih_deg + Cmde*de_deg +...
%     Cmalpha_dot*((alpha_dot*c)/(2*V)) + Cmq_dot*((q_dot*c^2)/(4*V^2)) ;
%% Momento de rolamento
Clbeta = -3.66e-3;
Clp = -0.661;
Clr = 0.144;
Clda = -2.87e-3;
Cldr = 6.76e-4;
Clbeta_dot = -0.263;
Clp_dot = -6.37e-3;
Clr_dot = 8.77e-3;

Cl = Clbeta*beta_deg + Clp*((p_rad_s*b)/(2*V)) + Clr*((r_rad_s*b)/(2*V)) +...
    Clda*da_deg + Cldr*dr_deg;
% Cl = Cl_beta*beta_deg + Clp*((p_rad_s*b)/(2*V)) + Clr*((r_rad_s*b)/(2*V)) +...
%     Clda*da_deg + Cldr*dr_deg +...
%     Clbeta_dot*((beta_dot*b)/(2*V)) + Clp_dot*((p_dot*b^2)/(4*V^2)) + Clr_dot*((r_dot*b^2)/(4*V^2));
%% Momento de guinada
Cnbeta = 5.06e-3;
Cnp = 0.0219; 
Cnr = -0.634;
Cnda = 0;
Cndr = -3.26e-3;
Cnbeta_dot = 0.811;
Cnp_dot = 3.98e-3;
Cnr_dot = -0.0958;

Cn = Cnbeta*beta_deg + Cnp*((p_rad_s*b)/(2*V)) + Cnr*((r_rad_s*b)/(2*V)) +...
    Cnda*da_deg + Cndr*dr_deg;

% Cn = Cnbeta*beta_deg + Cnp*((p_rad_s*b)/(2*V)) + Cnr*((r_rad_s*b)/(2*V)) +...
%     Cnda*da_deg + Cndr*dr_deg +...
%     Cnbeta_dot*((beta_dot*b)/(2*V)) + Cnp_dot*((p_dot*b^2)/(4*V^2)) + Cnr_dot*((r_dot*b^2)/(4*V^2));
%%
H_m = X(5);
rho = ISA(H_m);
q_bar = 0.5*rho*V^2;

L = q_bar*S*CL;
D = q_bar*S*CD;
Y = q_bar*S*CY;

La = q_bar*S*b*Cl;
Ma = q_bar*S*c*Cm;
Na = q_bar*S*b*Cn;

Faero_b =Cmat(2,alpha_rad)*Cmat(3,-beta_rad)*[-D; -Y;-L];

Maero_O_b = [La; Ma; Na];

Yaero = [L
    D
    Y
    La
    Ma
    Na];
end


