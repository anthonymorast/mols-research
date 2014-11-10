%Find Reduced Latin Squares Order 4
%
% Goes through all squares of order 4 and find the ones having first row 1
% 2 3 4 and put them in a seperate file. 

fid = fopen('4_latin.dat');
fid_write = fopen('reduced_order4.dat','w');

while ~feof(fid)
   A = fscanf(fid, '%u', [4 4]);
   A = A';
   if isequal(A, '')
      continue; 
   end
   
   if (A(1,1) == 1 && A(1,2) == 2 && A(1,3) == 3 && A(1,4) == 4)  
      fprintf(fid_write, '%u %u %u %u ', A);
      fprintf(fid_write, '\n');
   end
end

fclose('all');