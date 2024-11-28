function u=cubic_transition(t,ti,tf,ui,uli,uf,ulf)
% valor da função u
% valor do tempo t
M = [1 ti ti^2 ti^3
    0 1 2*ti 3*ti^2
    1 tf tf^2 tf^3
    0 1 2*tf 3*tf^2];

b = [ui uli uf ulf].';
% resolção do sistema
a = M\b;
% interpolação no tempo
u = a.'*[1 t t^2 t^3].';