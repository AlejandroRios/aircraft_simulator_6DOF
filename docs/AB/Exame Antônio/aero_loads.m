function [Faero_b,Maero_O_b,Yaero] = aero_loads(X,U)
global aircraft
V                         = X(1);
alpha_deg                 = X(2);
q_deg_s                   = X(3);
H                         = X(5);
beta_deg                  = X(7);
p_deg_s                   = X(9);
r_deg_s                   = X(10);
ih                        = U(3);
delta_e                   = U(4);
delta_a                   = U(5);
delta_r                   = U(6);

%% -----------------------------Aeronave----------------------------------%
b                         = aircraft.b;
c                         = aircraft.c;
S                         = aircraft.S;

%% ----------------------------Conversões---------------------------------%
q_rad_s                   = degtorad(q_deg_s);
p_rad_s                   = degtorad(p_deg_s);
r_rad_s                   = degtorad(r_deg_s);

%% ----------------------Matriz de Transformação--------------------------%
C_alpha                   = Cmat(2,degtorad(alpha_deg));
C_beta                    = Cmat(3,degtorad(beta_deg));
C_ba                      = C_alpha*C_beta;

%% -----------------------------Atmosfera---------------------------------% 
[rho,~,~,~]               = ISA(H);
q_bar                     = 0.5*rho*V^2;

%% -----------------------------Sustentação-------------------------------%
CL0                       = 0.308;
CLa                       = 0.133;
CLq                       = 16.7;
CLih                      = 0.0194;
CLde                      = 0.00895;

CL                        = CL0+CLa*alpha_deg+CLq*((q_rad_s*c)/(2*V))+CLih*ih+CLde*delta_e;
La                        = q_bar*S*CL;

%% --------------------------------Arrasto--------------------------------%
CD0                       = 0.02207;
CDa                       = 0.00271;
CDa2                      = 0.000603;
CDq2                      = 35.904;
CDb2                      = 0.00016;
CDp2                      = 0.5167;
CDr2                      = 0.5738;
CDih                      = -0.00042;
CDih2                     = 0.000134;
CDde2                     = 4.61D-5;
CDda2                     = 3D-5;
CDdr2                     = 1.81D-5;

CD                        = CD0+CDa*alpha_deg+CDa2*alpha_deg^2+CDq2*((q_rad_s*c)/(2*V))^2+CDb2*beta_deg^2+CDp2*((p_rad_s*b)/(2*V))^2+CDr2*((r_rad_s*b)/(2*V))^2+CDih*ih+CDih2*ih^2+CDde2*delta_e^2+CDda2*delta_a^2+CDdr2*delta_r^2;
D                         = q_bar*S*CD;

%% --------------------------------Lateral--------------------------------%
Cyb                       = 0.0228;
Cyp                       = 0.084;
Cyr                       = -1.21;
Cyda                      = 2.36D-4;
Cydr                      = -5.75D-3;

CY                        = Cyb*beta_deg+Cyp*((p_rad_s*b)/(2*V))+Cyr*((r_rad_s*b)/(2*V))+Cyda*delta_a+Cydr*delta_r;
Y                         = q_bar*S*CY;

%% --------------------------Momento de Arfagem---------------------------%
CM0                       = 0.017;
CMa                       = -0.0402;
CMq                       = -57;
CMih                      = -0.0935;
CMde                      = -0.0448;

CM                        = CM0+CMa*alpha_deg+CMq*((q_rad_s*c)/(2*V))+CMih*ih+CMde*delta_e; 
M                         = q_bar*S*c*CM;

%% -------------------------Momento de Rolamento--------------------------%
Clb                       = -3.66D-3;
Clp                       = -0.661;
Clr                       = 0.144;
Clda                      = -2.87D-3;
Cldr                      = 6.76D-4;

Cl                        = Clb*beta_deg+Clp*((p_rad_s*b)/(2*V))+Clr*((r_rad_s*b)/(2*V))+Clda*delta_a+Cldr*delta_r;
L                         = q_bar*S*b*Cl;

%% ---------------------------Momento de Guinada--------------------------%
Cnb                       = 5.06D-3;
Cnp                       = 0.0219;
Cnr                       = -0.634;
Cnda                      = 0;
Cndr                      = -3.26D-3;

Cn                        = Cnb*beta_deg+Cnp*((p_rad_s*b)/(2*V))+Cnr*((r_rad_s*b)/(2*V))+Cnda*delta_a+Cndr*delta_r;
N                         = q_bar*S*b*Cn;

%% --------------------------Forças Aerodinâmicas-------------------------%
Faero_b                   = C_ba*[-D;-Y;-La];

%% -------------------------Momentos Aerodinâmicos------------------------%
Maero_O_b                 = [L;M;N];

%% ---------------------------------Saidas--------------------------------%
Yaero                     = [CL; CD; CY; CM; Cl; Cn]; 

end