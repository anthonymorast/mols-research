function B = SRC_permute ( D, sigma )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Anthony Morast
% Undergraduate Research -- Brent Deschamp
%
% This function takes a D matrix of size 3 x n*n and a vecotr sigma which
% is the current permuatation. The rows of the D matrix are permuted in
% accordance to the sigma vector. The newly permuted D matrix is passed
% back to the overlord function.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B(1,:) = D(sigma(1),:);
B(2,:) = D(sigma(2),:);
B(3,:) = D(sigma(3),:);