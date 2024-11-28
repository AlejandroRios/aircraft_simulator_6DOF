function [Fprop_b,Mprop_O_b,Yprop] = prop_loads(~,U)
Tle                     = U(1);
Tre                     = U(2);

%% ---------------------------Parâmetros do Motor-------------------------%
ile                     = 2;
ire                     = 2;
tau_le                  = 1.5;
tau_re                  = -1.5;

%% -------------------------Matriz de Transformação-----------------------%
Clb_e                   = Cmat(2,degtorad(ile)).'*Cmat(3,degtorad(tau_le)).';
Crb_e                   = Cmat(2,degtorad(ire)).'*Cmat(3,degtorad(tau_re)).';

%% -------------------Ponto de aplicação da Força Propulsiva--------------%
rle_b                   = [4.899; -5.064; 1.435];
rre_b                   = [4.899; 5.064; 1.435];

%% ---------------------------Decomposição das Forças---------------------%
Txle                    = Tle*cosd(ile);
Tyle                    = Tle*sind(tau_le);
Tzle                    = Tle*sind(ile);
Txre                    = Tre*cosd(ire);
Tyre                    = Tre*sind(tau_re);
Tzre                    = Tre*sind(ire);
Tl                      = [Txle; Tyle; Tzle];
Tr                      = [Txre; Tyre; Tzre];
%% -----------------------------Forças Propulsivas------------------------%
F_esq                   = Clb_e*Tl;
F_dir                   = Crb_e*Tr;
Fprop_b                 = F_esq+F_dir;  

%% ----------------------------Momentos Propulsivos-----------------------%
M_esq                   = F_esq.*rle_b;
M_dir                   = F_dir.*rre_b;
Mprop_O_b               = M_dir-M_esq;

%% ---------------------------------Saidas--------------------------------%
Yprop                   = [F_esq; F_dir; M_esq; M_dir; Tl; Tr];
end