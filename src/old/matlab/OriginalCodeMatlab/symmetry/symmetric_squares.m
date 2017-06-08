function symmetric_squares ( filename, order )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Senior Research - Symmetry
% Anthony Morast
%
% This prorgram opens a file for input and a file to be written to. It
% reads in Latin squares from the read file and determines if the transpose
% is equivalent to the original. If it is they squares are symmetric, thus
% the rows and columns being permuted (placing the 1 in the (1,1) entry)
% will not create a different normailized Latin Square
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%open read file
fid = fopen ( filename );

%open read file
filename_write = 'symmetric_squares';
filename_write = strcat(filename_write,'_');
filename_write = strcat(filename_write, int2str(order));
filename_write = strcat(filename_write, '.dat');
fid2 = fopen ( filename_write, 'w' );


while ( ~feof(fid) )
    %read in ls
    A = fscanf(fid, '%u', [order,order]);
    
    for i=1:order
        for j=1:order
            A(i,j) = A(i,j) + 1;
        end
    end
      
    %create format
    format = '%u';
    for i=1:order - 1 
        format = strcat(format, ' %u');
    end
    format = [format, ' '];
    
    %if not last square in file
    if size(A) ~= 0
        %determine if square is symmetric
        B = A';
        if isequal(A,B) == 1
            fprintf( fid2, format, A );
            fprintf ( fid2, '\n');
        end
    end
    
end
                   
fclose ('all');
