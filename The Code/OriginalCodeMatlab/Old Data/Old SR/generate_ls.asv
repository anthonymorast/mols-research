function  generate_ls ( A )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Anthony Morast, Senior Research
% 
% This program takes a normalized latin square and permutes the rows and columns to generate 
% latin squares of a particular order. The permutations of n are generated via a c++ program.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%open file containing all permutations of 1,2,3, and 4
fid = fopen ( '4_perm.txt' );

%Get size of latin square
[m,n] = size(A);

%create a second instance of the latin square passed into the function
B = A;

%read in first value in file, # permutations
fscanf (fid,'%d', 1);

%initialize k
k = 0;

%while not end of file, i.e. factorial of latin square order 
while ( k < factorial(m) )

  %read permutation into sigma vector
  sigma = fscanf (fid,'%d', m);
 
  for i=1:n
    %permute columns
    %get value of comlumn for D
    j = sigma(i);
    %move column i to column j
    D(:,j) = B(:,i);
    % D is the permuted matrix 
  end

  %permute rows, D as input
  permute_rows(D);

  %increment k value
  k = k + 1;
end
%end 