%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This m file reads in the normalized latin squares of order 5 that have
% aready been generated. The symbols of these latin squares are permuted
% all possible ways and, if not already inside a file, are written to a
% file after they are normailized.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid3 = fopen ( 'missing_ls.txt' );

while ~feof(fid3)
    A = fscanf(fid3, '%u', [5 5]);
    if size(A) ~= 0
         permu_iso_2(A);
    end
end

%eq_ls;
    
fclose('all');