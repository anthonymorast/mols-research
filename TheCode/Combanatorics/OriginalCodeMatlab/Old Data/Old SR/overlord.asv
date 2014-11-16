function overlord

%open file to write latin squares that 'work' together
fid2 = fopen('latin_works_3.txt','w');

%loop through all latin squares of order 4
for i=1:576
    %open file containing all latin squares
    fid = fopen('LS4_backup.dat');

    %%%%%%%Read in dummies
    for k=1:(i-1)
        dummy = fscanf(fid,'%u',[4 4]);
    end
    
    %Read in first LS
    C = fscanf(fid,'%u',[4 4]);
    C =  C';
    for j=i+1:576
        %read in latin square to be compared
        D = fscanf(fid,'%u',[4 4]);
        D=D';
        %if the latin squares work together write their postions in the latin square file to another file
        if ( check_latin_square(C,D) == 1 )
             fprintf(fid2, '%u ', i);
             fprintf(fid2, '%u', j);
             fprintf(fid2,'\n');
        end
    end
    %close file containing order 4 latin squares
    fclose(fid);

end

%close file
fclose(fid2);
        