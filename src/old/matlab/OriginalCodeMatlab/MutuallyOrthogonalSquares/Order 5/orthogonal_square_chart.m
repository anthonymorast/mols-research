% Orthogonal Square Chart
%
% Creates a file with the sets of orthogonal squares.. It finds the sets of
% two first then reads the file in checking each pair with the list of the
% squares in the vector.

%create vector of all reduced squares
fid = fopen('reduced_squares_o5.dat');

reduced_squares = zeros(5,5,0);
j  = 0;
while ~feof(fid)
   A = fscanf(fid, '%u', [5 5]);
   if ~isequal(A,'') 
       reduced_squares = cat(3,reduced_squares,A);
       j = j + 1;
   end
end
[m,n,o] = size(reduced_squares);

fid_chart = fopen('mutually_orthogonal_squaresO5.dat','w');

%check each square in the vector with each other square in the vector
for i=1:o
   currSquare = reduced_squares(:,:,i);
   for j=i+1:o
      checkSquare = reduced_squares(:,:,j);
      %write the pairs to a file
      if (check_latin_square(currSquare,checkSquare))
          fprintf(fid_chart,'%u %u \n', i, j);
      end
   end
end

%close file contianing sets of squares
fclose('all');
