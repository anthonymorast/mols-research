fid = fopen('mckay_nls5.dat');
fid1 = fopen('nlsO5.dat', 'wt');

while (~feof(fid))
   A = fscanf(fid, '%u', [5 5]);
   
   for i=1:5
       for j=1:5
           A(i,j) = A(i,j) + 1;
       end
   end
   fprintf(fid1,'%u %u %u %u %u ', A);
   fprintf(fid1, '\n');
end 