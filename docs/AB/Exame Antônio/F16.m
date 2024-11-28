clear all
close all
clc

global g aircraft
% O modelo do F-16 é construído no sistema de unidades inglesas:
m2ft = 1/0.3048;
ft2m = 1/m2ft;
g = 9.80665*m2ft;

% Outras conversões:
lb2kg = 0.45359237;
kg2lb = 1/lb2kg;

slug2kg = g*lb2kg;
kg2slug = 1/slug2kg;

deg2rad = pi/180;
rad2deg = 1/deg2rad;

%--------------------------------------------------------------------------
% Dados geométricos:
b = 30;     % Envergadura [ft]
S = 300;    % Área de referência [ft²]
c = 11.32;  % Corda média aerodinâmica [ft]

%--------------------------------------------------------------------------
% Dados de inércia:
W = 20500;
% Massa [slug]:
m = W/g;
% Momentos de inércia [slug·ft²]:
Ixx = 9496;
Iyy = 55814;
Izz = 63100;
% Produtos de inércia [slug·ft²]:
Ixy = 0;
Ixz = 982;
Iyz = 0;
J = [Ixx -Ixy -Ixz
    -Ixy Iyy -Iyz
    -Ixz -Iyz Izz];
% CG nominal: 0.35c
xCG = -0.05*c;
yCG = 0;
zCG = 0;
rC_b = [xCG yCG zCG].';
r_pilot_b = [15; 0; 0];
%--------------------------------------------------------------------------
% Quantidade de movimento angular do motor [slug·ft²/s]:
hex = 160;

aircraft = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'hex',hex,'r_pilot_b',r_pilot_b);

% Testes:
alpha_rad = 0.5;
beta_rad = -0.2;
C_ba = Cmat(2,alpha_rad)*Cmat(3,-beta_rad);
X = [C_ba*[500; 0; 0]
    0.7
    -0.8
    0.9
    1000
    900
    -10000
    -1
    1
    -1
    90];
U = [0.9
    20
    -15
    -20];

Xdot = dynamics(0,X,U)