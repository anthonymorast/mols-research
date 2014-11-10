function is_latin (order)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determines whether or not a latin square read in from
% a file is a latin square or not.
%
% If it is print it, if not do nothing.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%open file(s)
fid = fopen('5iso_permutation.dat');
fid2 = fopen('normalized_ls5.dat','w');


%check if this is a latin square
i = 0;
count = 0;
while (i < factorial(order))
    %read in latin square
    A = fscanf(fid,'%u', [order,order]);
    
    %check rows
    for j=1:order
        for k=1:order
            for n=k+1:order
                if A(j,k) == A(j,n);
                    count = count + 1;
                    break
                end
            end
        end
    end 
    
    %check columns
    count = 0;   
    for j=1:order
        for k=1:order
            for n=k+1:order
                if A(k,j) == A(n,j);
                    count = count + 1;
                    break
                end
            end
        end
    end 
    
    %write to file if it is a ls
    if count == 0
        fprintf(fid2, '%u %u %u %u %u ', A);
        fprintf(fid2, '\n');
    end
    
    %increment i
    i = i+1;
end



