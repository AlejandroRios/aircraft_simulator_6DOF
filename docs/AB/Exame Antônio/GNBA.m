global g aircraft
%% ------------------------Parâmetros geométricos-------------------------%
b                   = 32.757;     % Envergadura [m]
S                   = 116;        % Área de referência [m²]
c                   = 3.862;      % Corda média aerodinâmica [m]

%% ------------------------------Inércia----------------------------------%
m                   = 55788;      % Massa [Kg]
W                   = m*g;        % Peso  [Kg]
Ixx                 = 821466;
Iyy                 = 3343669;
Izz                 = 4056813;
Ixy                 = 0;
Ixz                 = 178919;
Iyz                 = 0;
J                   = [Ixx -Ixy -Ixz
                       -Ixy Iyy -Iyz
                       -Ixz -Iyz Izz];
xCG                 = 0;
yCG                 = 0;
zCG                 = 0;
rC_b                = [xCG yCG zCG].';
r_pilot_b           = [0; 0; 0];

%% -------------------------------Saída-----------------------------------%
aircraft            = struct('m',m,'J_O_b',J,'rC_b',rC_b,'b',b,'S',S,'c',c,'r_pilot_b',r_pilot_b);
