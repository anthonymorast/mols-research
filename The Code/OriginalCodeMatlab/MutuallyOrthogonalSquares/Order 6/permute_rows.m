function  D =permute_rows(A)

%open file contating permutations of 1, 2, 3
fid1 = fopen( '5_perm.txt' );

%open file for writing latin squares of order 4
fid2 = fopen ('latin_squares_order6.dat','a');

%get size of matrix passed in
[m,n] = size(A);

%read in first line of permutation file
fscanf ( fid1, '%u', 1 );

%create instance of passed in latin square, initialize k value
D = A;
k = 0;

%loop until all permutations are found
while ( k < factorial(m-1) )
    %read values into sigma vector to determine new row postions	
    sigma = fscanf ( fid1, '%u', n-1 );

    %permute rows
    for i=1:n-1    
      j = sigma(i);
      %move row j+1 to row i+1
      D(j+1,:) = A(i+1,:);
    end

    %print latin square to file
    fprintf(fid2, '%u %u %u %u %u %u ', D);

    %increment k value
    k = k + 1;
    %print new line
    fprintf(fid2, '\n');
end

%close files
fclose(fid1);
fclose(fid2);