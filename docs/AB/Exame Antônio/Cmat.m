function C = Cmat(n,angle_rad)

nn = n;
if length(nn)>1, nn=4; end

switch nn %Escolher eixo de rotação
    case 1 %Rotação em gama
        C = [1 0 0
            0 cos(angle_rad) sin(angle_rad)
            0 -sin(angle_rad) cos(angle_rad)];
    case 2 %Rotação em theta
        C = [cos(angle_rad) 0 -sin(angle_rad)
            0 1 0
            sin(angle_rad) 0 cos(angle_rad)];
    case 3 %Rotação em phi
        C = [cos(angle_rad) sin(angle_rad) 0
            -sin(angle_rad) cos(angle_rad) 0
            0 0 1];
    otherwise %Caso geral
        %Criar vetor unitário de n
        n = n/norm(n);
        %Matriz de transformação dado n e angulo
        C = (1-cos(angle_rad))*(n*n.')+cos(angle_rad)*eye(3)+-sin(angle_rad)*skew(n);
end
end

