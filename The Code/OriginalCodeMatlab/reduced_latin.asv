function reduced_latin

%open list of order four latin squares
fid = fopen('LS4_backup.dat');
%open file to store reduced latin squares.
fid2 = fopen('reduced_ls.dat', 'a');

i = 0;
%while not all latin squares are read, read and check if reduced
while ( i < 576 )
    %read in latin square
    A = fscanf(fid, '%u', [1 16]);
    %check if the square is reduced.
    if ( A(1) == 1 && A(2) == 2 && A(3) == 3 && A(4) == 4 )
        %print to file if reduced.
        fprintf(fid2, '%u %u %u %u ', A); 
        fprintf(fid2,'\n');
    end
    %increment counter
    i = i + 1;
end

%close files
fclose('all');

