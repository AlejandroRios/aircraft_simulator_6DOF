function X = state_vec(x,trim_par);
%x =[V alpha q theta H beta phi p r|T ih da dr]'; %Incógnitas
%X = [V alpha q theta H x beta phi p r psi y]
X           = zeros(12,1);  %Matriz de estados
X(1)        = trim_par.V;
X(2:4)      = x(2:4);
X(5)        = trim_par.H;
X(7:10)     = x(6:9);

end