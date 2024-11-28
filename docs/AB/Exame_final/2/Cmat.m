function C = Cmat(n,angle_rad)

 nn = n;
 if length(nn)>1, nn=4; end
 
switch nn
    case 1
        C = [1 0 0
            0 cos(angle_rad) sin(angle_rad)
            0 -sin(angle_rad) cos(angle_rad)];
    case 2
        C = [cos(angle_rad) 0 -sin(angle_rad)
            0 1 0
            sin(angle_rad) 0 cos(angle_rad)];
    case 3
        C = [cos(angle_rad) sin(angle_rad) 0
            -sin(angle_rad) cos(angle_rad) 0
            0 0 1];
    otherwise
        n = n/norm(n); % vetor unitario n 

        C = 1-cos(angle_rad)*(n*n.') +...
            cos(angle_rad)*eye(3)    +...
            -sin(angle_rad)*skew(n);
end
