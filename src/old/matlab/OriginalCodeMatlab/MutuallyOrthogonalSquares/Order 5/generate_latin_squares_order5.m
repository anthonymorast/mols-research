%Generate Latin Squares Order 5
%
% Generates all the Latin squares of order 5 by passing in each normalized
% Latin square one by one and permuting the rows and columns. Should create
% a file with 161280 squares.

%open file containing normalized Latin squares of order 5
fid = fopen('normalized_ls_order5.dat');

while (~feof(fid))
    A = fscanf(fid, '%u', [5 5]);

    %pass square into generate_ls
    generate_ls(A);
end 