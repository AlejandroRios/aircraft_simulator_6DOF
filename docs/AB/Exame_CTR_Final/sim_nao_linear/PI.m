function f = PI(K,A,B,C,F,G,H,Q,R,r0,V)
    Ac = A - B*K*C;
    Bc = G - B*K*F;
    rmax = max(real(eig(Ac)));

    if rmax<0
        x_stat = - (Ac\Bc)*r0;
        e_stat = (1 + H*(Ac\Bc)) * r0;
        P = lyap(Ac',Q+C'*K'*R*K*C);
        f=0.5*trace(P*(x_stat*x_stat')) + 0.5*e_stat'*V*e_stat;

    else
        f = 1e20;
    end
end
