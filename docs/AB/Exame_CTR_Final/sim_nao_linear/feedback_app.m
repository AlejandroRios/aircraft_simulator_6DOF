function [Xdot1,Y1,U] = feedback_app(t,X_eq,U_eq,Y_eq)

U = U_eq;
% Referencias para rastreio
dref =0;
Vref = 500;

[Xdot1,Y1] = dynamics(t,X_eq,U_eq);

t_i = 0; % t para inicializar PA Approach

if t > t_i

if Y1(5) >= 70
%% Malha longitudinal (Aproximação)
% Ganhos correspondentes a aproximação vindos da sim. linear
Kalpha =  39.0483; 
Ktheta = -38.9689;
Kq =  -4.0686;
Ked =  0.0116;
Kied =  0.0033;
Kev =    -0.0396   ;
Kiev =  -0.0004;

% Lei de controle realimentação alpha, theta e q
U_1_p = (Y1(2)-Y_eq(2))*(-Kalpha) + (Y1(4)-Y_eq(4))*(-Ktheta)  + (Y1(3)-Y_eq(3))*(-Kq) ;

d=X_eq(19);
V = Y1(1);

% Erro d e V
erro_d = (dref-d);
erro_V = (Vref-V);

%integral do erro
epsilon_d = erro_d;
epsilon_v = erro_V;

% Realimentação proporcional e integral para profundor e throttle
U_2_p = erro_d*(-Ked) + epsilon_d*(-Kied);
U_2_t = erro_V*(-Kev) + epsilon_v*(-Kiev);

% Soma das contribuições do feedback
U_fb_p =U_2_p+U_1_p;
U_fb_t =U_2_t;

% delta p e delta t
delta_p = Xdot1(17);
delta_t = Xdot1(18);

% Modelagem com atuador do profundor e manete
U_fb_p = (20.2)*(-delta_p+(U_fb_p));
U(2) = U(2)+delta_p;

U_fb_t =(-delta_t+(U_fb_t));
U(1) = U(1)+delta_t;

% Limitado de profundor
if U(2) > 30
    U(2) = 30;
elseif U(2) < -30
    U(2) = -30;
else
    U(2) = U(2);
end

% Limitado de manete
if U(1) > 1
    U(1) = 1;
elseif U(1) < 0
    U(1) = 0;
else
    U(1) = U(1);
end
%% Malha latero direcional
% Ganhos correspondentes a PA de direção obtidos de sim. linear
Kiphi = 15.6208  ;
Ker = 2.5121;
Kp =  -2.1879  ;
Kephi = 14.8967;
Kpsi = 3;

% Lei de controle realimentação de p
U_1_a = (Y1(9)-Y_eq(9))*(-Kp);

% Referencias para rastreio
Vt = 500;
r_ref = 0;
psi_ref = 0;

r = Y1(10);
phi = Y1(8);
psi = Y1(11);

% Erro de psi
erro_psi = (psi_ref-psi);

tau_psi = Vt/(Kpsi*9.81);

% Phi de referencia para rastreio
phi_ref = (Vt/(tau_psi*9.81))*erro_psi;

% Erro de phi e r
erro_phi = (phi_ref - phi);
erro_r = (r_ref -r);
% Integral de erro de phi
epsilon_phi = erro_phi;

% Realimentação proporcional e integral para aileron e rudder
U_2_a = erro_phi*(-Kephi) + epsilon_phi*(-Kiphi);
U_2_r = erro_r*(-Ker);

U_fb_a =U_2_a+U_1_a;
U_fb_r =U_2_r;

% delta a 
delta_a = Xdot1(23);
delta_r = Xdot1(24);

% Modelagem do atuador do aileron e leme
U_fb_a = (20.2)*(-delta_a+(U_fb_a));
U(3) = U(3)+delta_a;

U_fb_r = (20.2)*(-delta_r+(U_fb_r));
U(4) = U(4)+delta_r;

% Limitador de aileron
if U(3) > 30
    U(3) = 30;
elseif U(3) < -30
    U(3) = -30;
else
    U(3) = U(3);
end

% Limitador de leme
if U(4) > 30
    U(4) = 30;
elseif U(4) < -30
    U(4) = -30;
else
    U(4) = U(4);
end

%% Calculo da dinâmica de malha fechada
[Xdot1,~] = dynamics(t,X_eq,U);

%Erro d e V
Xdot1(15) = erro_d;
Xdot1(16) = erro_V;
%Erro phi e r
Xdot1(21) = erro_phi;
Xdot1(22) = erro_r;

%Epsilon
Epsilon_d = Xdot1(15);
Epsilon_V = Xdot1(16);

%delta e e t
Xdot1(17) = U_fb_p ;
Xdot1(18) = U_fb_t;
%delta a e t
Xdot1(23) = U_fb_a ;
Xdot1(24) = U_fb_r ;

elseif Y1(5) < 70
%% Malha longitudinal (Arredondamento)
% Ganhos correspondentes ao arredondamento vindos da sim. linear
Kalpha = 31.9284 ; 
Ktheta =  -32.0135    ;
Kq = -2.2443   ;
Keh =-0.0015  ;
Kieh =   -1.0970;
Kev =     -0.0471  ;
Kiev =     -0.0005    ;

% Lei de controle realimentação alpha, theta e q
U_1_p = (Y1(2)-Y_eq(2))*(-Kalpha) + (Y1(4)-Y_eq(4))*(-Ktheta)  + (Y1(3)-Y_eq(3))*(-Kq) ;

% Distância de referencia para arredondamento
href =X_eq(20) ;

H = Y1(5);
V = Y1(1);

% Erro H e V
erro_h = (H-href);
erro_V = (Vref-V);

%integral do erro H e V
epsilon_h = erro_h;
epsilon_v = erro_V;

% Realimentação proporcional e integral para profundor e throttle
U_2_p = erro_h*(-Keh) + epsilon_h*(-Kieh);
U_2_t = erro_V*(-Kev) + epsilon_v*(-Kiev);

% Soma das contribuições do feedback
U_fb_p =U_2_p+U_1_p;
U_fb_t =U_2_t;

% delta p e delta t
delta_p = Xdot1(17);
delta_t = Xdot1(18);

% Modelagem com atuador do profundor e manete
U_fb_p = (20.2)*(-delta_p+(U_fb_p));
U(2) = U(2)+delta_p;

% Limitado de profundor
if U(2) > 30
    U(2) = 30;
elseif U(2) < -30
    U(2) = -30;
else
    U(2) = U(2);
end

% Limitado de manete
if U(1) > 1
    U(1) = 1;
elseif U(1) < 0
    U(1) = 0;
else
    U(1) = U(1);
end
%% Malha latero direcional
% Ganhos correspondentes a PA de direção obtidos de sim. linear
Kiphi = 15.6208  ;
Ker = 2.5121;
Kp =  -2.1879  ;
Kephi = 14.8967;
Kpsi = 3;

% Lei de controle realimentação de p
U_1_a = (Y1(9)-Y_eq(9))*(-Kp);

if X_eq(8) > 63
    
% Referencias para rastreio
Vt = 500;
r_ref = 0;
psi_ref = -5.2;

r = Y1(10);
phi = Y1(8);
psi = Y1(11);

% Erro de psi
erro_psi = (psi_ref-psi);
tau_psi = Vt/(Kpsi*9.81);

% Phi de referencia para rastreio
phi_ref = (Vt/(tau_psi*9.81))*erro_psi;

% Erro de phi e r
erro_phi = (phi_ref - phi);
erro_r = (r_ref -r);

% Integral de erro de phi
epsilon_phi = erro_phi;

% Realimentação proporcional e integral para aileron e rudder
U_2_a = erro_phi*(-Kephi) + epsilon_phi*(-Kiphi);
U_2_r = erro_r*(-Ker);

U_fb_a =U_2_a+U_1_a;
U_fb_r =U_2_r;

% delta a e r
delta_a = Xdot1(23);
delta_r = Xdot1(24);

% Modelagem do atuador do aileron e leme
U_fb_a = (20.2)*(-delta_a+(U_fb_a));
U(3) = U(3)+delta_a;

U_fb_r = (20.2)*(-delta_r+(U_fb_r));
U(4) = U(4)+delta_r;

% Limitador de aileron
if U(3) > 30
    U(3) = 30;
elseif U(3) < -30
    U(3) = -30;
else
    U(3) = U(3);
end

% Limitador de leme
if U(4) > 30
    U(4) = 30;
elseif U(4) < -30
    U(4) = -30;
else
    U(4) = U(4);
end
%%
else        
% Referencias para rastreio
Vt = 500;
r_ref = 0;
psi_ref = 0;

r = Y1(10);
phi = Y1(8);
psi = Y1(11);

% Erro de psi
erro_psi = (psi_ref-psi);
tau_psi = Vt/(Kpsi*9.81);

% Phi de referencia para rastreio
phi_ref = (Vt/(tau_psi*9.81))*erro_psi;

% Erro de phi e r
erro_phi = (phi_ref - phi);
erro_r = (r_ref -r);

% Integral de erro de phi
epsilon_phi = erro_phi;

% Realimentação proporcional e integral para aileron e rudder
U_2_a = erro_phi*(-Kephi) + epsilon_phi*(-Kiphi);
U_2_r = erro_r*(-Ker);

U_fb_a =U_2_a+U_1_a;
U_fb_r =U_2_r;

% delta a e r
delta_a = Xdot1(23);
delta_r = Xdot1(24);

% Modelagem do atuador do aileron e leme
U_fb_a = (20.2)*(-delta_a+(U_fb_a));
U(3) = U(3)+delta_a;

U_fb_r = (20.2)*(-delta_r+(U_fb_r));
U(4) = U(4)+delta_r;

% Limitador de aileron
if U(3) > 30
    U(3) = 30;
elseif U(3) < -30
    U(3) = -30;
else
    U(3) = U(3);
end

% Limitador de leme
if U(4) > 30
    U(4) = 30;
elseif U(4) < -30
    U(4) = -30;
else
    U(4) = U(4);
end

end

% Calculo da dinâmica de malha fechada
[Xdot1,~] = dynamics(t,X_eq,U);

% Erro h e V
Xdot1(14) = erro_h;
Xdot1(16) = erro_V;

%Erro phi e r
Xdot1(21) = erro_phi;
Xdot1(22) = erro_r;

%Epsilon
epsilon_h = Xdot1(14);
Epsilon_V = Xdot1(16);

%delta e e t
Xdot1(17) = U_fb_p ;
Xdot1(18) = U_fb_t;   

%delta a e t
Xdot1(23) = U_fb_a ;
Xdot1(24) = U_fb_r ;
end    
end

