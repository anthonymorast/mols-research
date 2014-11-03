%Overlord Generate Latin Squares
%
% This script will read in all normalized Latin squares and permute the
% rows and columns to produce all Latin squares of order 6.

fid = fopen('normalized_ls_order6.dat');
i=1;
while ~feof(fid)
   A = fscanf(fid, '%u', [6 6]);
   generate_ls(A);
   i=i+1;
   disp(i);
end