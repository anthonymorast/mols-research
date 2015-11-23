%open files
fid_6 = fopen('setOf6.dat');
fid_10 = fopen('setOf10.dat');
fid_40 = fopen('setOf40.dat');
fid_out = fopen('chart.dat','wt');

%read in nls 
squares6 = zeros(5,5,0);
squares10 = zeros(5,5,0);
squares40 = zeros(5,5,0);

while (~feof(fid_6))
   A = fscanf(fid_6,'%u',[5 5]); 
   squares6 = cat(3, squares6, A);   
end

while (~feof(fid_10))
   A = fscanf(fid_10,'%u',[5 5]); 
   squares10 = cat(3, squares10, A);
end

while (~feof(fid_40))
    A = fscanf(fid_40,'%u',[5 5]);
    squares40 = cat(3, squares40, A);
end

%create filename base
filename = 'perm_class';

fprintf(fid_out, '%s      %s', 'Isotopy Class', 'Permute Class');
fprintf(fid_out,'\n');
for i=1:16
   currentfile = strcat(filename, int2str(i));
   currentfile = strcat(currentfile, '.dat');
   fid = fopen(currentfile);
   
   %read in 'permutation class' squares and compare with nls'
   while (~feof(fid))
       A = fscanf(fid, '%u', [5 5]);
       contains40 = 0;
       contains10 = 0;
       contains6 = 0;
       
       %see if it's one of the 6
       for j=1:6
           if (isequal(squares6(:,:,j),A))
               contains6 = 1;
               break;
           end
       end
       
       for j=1:10
           if (isequal(squares10(:,:,j),A))
                contains10 = 1;
                break;
           end
       end
       
       for j=1:40
           if (isequal(squares40(:,:,j),A))
               contains40 = 1;
               break;
           end
       end
       
       if contains6
           fprintf(fid_out, '%u                  %u', 6, i);
           fprintf(fid_out, '\n');
       elseif contains10
           fprintf(fid_out, '%u                 %u', 10, i);
           fprintf(fid_out, '\n');
       elseif contains40
           fprintf(fid_out, '%u                 %u', 40, i);
           fprintf(fid_out, '\n');
       end
   end
end

%close files
fclose('all');