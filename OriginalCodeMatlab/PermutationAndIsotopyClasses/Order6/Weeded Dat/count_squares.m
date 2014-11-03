%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reads in a bunch of files and adds the counts together.

filename = 'isotopyclass';
total = 0;
fid2 = fopen('normalized_ls_order6.dat', 'wt');
b = zeros(0);

for i=1:22
   %create file name
   this_file = strcat(filename,int2str(i));
   this_file = strcat(this_file, '-new');
   this_file = strcat(this_file, '.dat');

   fid = fopen(this_file);
   k=1;
   
   %count
   while (~feof(fid))
       A = fscanf (fid, '%u', [6 6]);
       if ( A == A')
          b(k) = i;
          k = k + 1;
       end
       total = total +1;
       fprintf(fid2, '%u %u %u %u %u %u ', A);
       fprintf(fid2,'\n');
   end
   
   fclose(fid);
end

disp(total);

i = size(b);

fclose('all');