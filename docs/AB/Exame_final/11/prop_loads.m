function [Fprop_b,Mprop_O_b,Yprop] = prop_loads(X,U)

global aircraft

%Yprop - variaveis de saida importantes (ex. tra??o, derivada no tempo da potencia, etc.)
% power = X(13);


H_m = X(5);
V = X(1);
[~,~,~,a] = ISA(H_m);
M = V/a;

% No sistema do corpo:
%% Força e momento motor esquerdo
ile_rad = degtorad(2);
taule_rad = degtorad(1.5);
rle_b = [4.899;-5.064;1.435];

Fx_le = U(1);
Fy_le = 0;
Fz_le = 0;

Mt_le = Cmat(2,ile_rad)*Cmat(3,taule_rad);
Fle = [Fx_le Fy_le Fz_le];
Fble = Mt_le*Fle';
Mble = Fble.*rle_b;
%% Força e momento motor direito
ire_rad = degtorad(2);
taure_rad = degtorad(-1.5);
rre_b = [4.899; 5.064;1.435];

Fx_re = U(2);
Fy_re = 0;
Fz_re = 0;

Mt_re = Cmat(2,ire_rad)*Cmat(3,taure_rad);
Fre = [Fx_re Fy_re Fz_re];
Fbre = Mt_re*Fre';
Mbre = Fbre.*rre_b;
%% Sumatorio de forças e momentos
T = Fble+Fbre;
Fprop_b = [T];

Mprop_O_b = Mble-Mbre;
Mprop_O_b =[Mprop_O_b];
Yprop = [Fble
    Fbre
    Mble
    Mbre];
