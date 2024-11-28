function f=stabK(K,A,B,C)
Ac=A-B*K*C;
f=max(real(eig(Ac)));