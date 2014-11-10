%
% This script reads in the latin squares of order 5 and if they are reduced
% prints them to another file 

fid = fopen('latin_squares_order5.dat');
fid_write = fopen ('reduced_squares_o5.dat', 'a');

while ~feof(fid)
    A = fscanf (fid, '%u', [5 5]);
    if (~isequal(A, []))
        A = A';
        if (A(1,1) == 1 && A(1,2) == 2 && A(1,3) == 3 && A(1,4) == 4 && A(1,5) == 5)
            fprintf (fid_write, '%u %u %u %u %u ', A);
            fprintf (fid_write, '\n');
        end
    end
end