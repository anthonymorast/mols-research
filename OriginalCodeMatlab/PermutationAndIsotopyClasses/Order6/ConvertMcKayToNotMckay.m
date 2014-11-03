fid = fopen('order6_reps.dat');
fid1 = fopen('order6_reps_notMckay.dat', 'wt');

while (~feof(fid))
   A = fscanf(fid, '%u', [6 6]);
   
   for i=1:6
       for j=1:6
           A(i,j) = A(i,j) + 1;
       end
   end
   fprintf(fid1,'%u %u %u %u %u %u ', A);
   fprintf(fid1, '\n');
end 