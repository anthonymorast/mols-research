fid = fopen('reduced_order4.dat');

i = 1;
while (~feof(fid))
   i
   A = fscanf(fid, '%u', [4 4])
   i = i + 1;
   pause
end

fclose('all');