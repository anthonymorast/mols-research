function main_class_overlord ( n )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Anthony Morast
% Undergraduate Research -- Brent Deschamp
%
% This program reads in the main class representatives of order n ( the
% parameter to the function ). The representatives are stored as a 3D
% array. The first index holds n*n values to represent the rows of a Latin
% Square, the second holds n*n values to represent columns, and the third
% holds the symbols ( numbers ) that populate the latin square. 
%
% The three indices are permuted to create new latin squares. After each
% permutation, 6 per representative, the new latin squares have their
% symbols permuted to create all other latin squares. This handles the
% identity case as the first permutation in the list of permutations is
% always the indentity permutation in out files. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize D matrix
D = zeros(3,n*n);

%open file(s)
fid = fopen('main_class1_weeded.dat');
k = 1;
%loop through all main class representatives in the file
while ( ~feof(fid) )
    display(1)
    %read in main class representative and transpose
    A = fscanf(fid, '%u', [n n]);
    A = A';
    %check if the file has not read "too far"
    if size(A) ~= 0
        %create D array
        for i=1:n
            for j=1:n
                D(1,(n*i) - (n-j)) = i;
                D(2,(n*i) - (n-j)) = j;
                D(3,(n*i) - (n-j)) = A(i,j);
            end
        end
        display(D)
        %permute rows/cols/symbols via D array in 3! ways
        perm = 1;
        fid2 = fopen ('3_perm.txt');
        a = fscanf(fid2,'%u',1);
        
        %create file name used in the isotopy permutation, one file for
        %each main class representative
        filename = 'main_class_';
        filename2 = filename;
        filename = strcat(filename ,int2str(k));
        filename = strcat(filename,'.dat');
        
        while ( perm <= factorial(a) )
            %create dummy of D to be permuted.
            B = D;
            %read in permutation
            sigma = fscanf(fid2, '%u', a);
            %permute row/col/symbol
            B = SRC_permute(B, sigma);
            %put D = B back into latin square form  (L)
            for i=1:n
                for j=1:n
                    L(B(1,(n*i)-(n-j)),B(2,(n*i)-(n-j))) = B(3,(n*i)-(n-j));
                end
            end
            %permute symbols of the new Latin Square and write permutations
            %to file
            permu_iso_2(L, filename);
            perm = perm + 1;
            eq_ls(filename2,k);
        end
    end
    k = k + 1;
end

fclose('all');
        
        
        
        
        
        
        
        

