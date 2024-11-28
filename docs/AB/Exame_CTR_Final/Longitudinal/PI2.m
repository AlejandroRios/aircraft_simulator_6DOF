function f = PI2(K,A,B,C,F,G,H,Q,R,r0,V,q)

K = [0,0,0,K(1),K(2),0,0;
    K(3),K(4),K(5),0,0,K(6),K(7)];

    Ac = A - B*K*C;
    Bc = G - B*K*F;
    rmax = max(real(eig(Ac)));

    % fazer K(1) = -0.01 (alpha)
    
    if rmax<0
    
        x_stat = - (Ac\Bc)*r0;
        e_stat = (1 + H*(Ac\Bc)) * r0;
    
    P0 = lyap(Ac',Q);
    P1 = lyap(Ac',P0);
    P  = lyap(Ac',q*2*P1+C'*K'*R*K*C);
    
  f=0.5*trace(P*(x_stat*x_stat')) + 0.5*e_stat'*V*e_stat;

    else
        f = 1e20;
 end