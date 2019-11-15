function gen_norm (B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generates all normalized latin squares by permuting
%    1 2 3 4   then normalizing this matrix to see if
%A = 2 1 4 3   it generates any of the 3 other normalized
%    3 4 1 2   latin squares of order 4.
%    4 3 2 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%open file containing permutation of order 4
fid = fopen ( '4_perm.txt' );

%read in number of permutations
n = fscanf(fid, '%u',1);

%open file for writing
fid2 = fopen('normls4.dat', 'w');

m=n;
i = 0;
while ( i < factorial(n) )
    A = B;
    %read in permutation
    sigma = fscanf ( fid, '%u', n);
    for j=2:n
        for k=2:n
            x = A(j,k);
            % 11/13/2019 - WTF were you doing? This is literally A(j,k) = sigma(x)
            if x == 1
                A(j,k) = sigma(1);
            elseif x == 2
                A(j,k) = sigma(2);
            elseif x == 3
                A(j,k) = sigma(3);
            else
                A(j,k) = sigma(4);
            end
        end
    end
    %print out latin square
    display(A);
    fprintf(fid2, '%u %u %u %u ', A);
    fprintf(fid2, '\n');
    i = i+1;
end
