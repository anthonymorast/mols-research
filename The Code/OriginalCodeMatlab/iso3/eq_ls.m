function eq_ls
% this function compares files of latin squares to find out which ones
%appear more than once. If the ls has not appeared it is output to another
%file.

%open file to write normalized ls to
fid2 = fopen('MM_permu2.dat', 'w');

fid = fopen('MM_permu.dat');

%read in first ls
A = fscanf(fid, '%u', [5 5]);
A = A';

B(1,:,:) = A;

while ( ~feof(fid) )
    %read in LS
    A = fscanf(fid, '%u', [5 5]);
    A = A';
    if size(A) ~= 0
        %get size of array
        [m, n, o] = size(B);
 
        %re-initialize flag to "not in array"
        flag = 0;
    
        %loop through array and compare with current LS
        for j=1:m
            for k=1:5
                C(k,:) = B(j,k,:);
            end
       
            %if they are equal,
            if isequal(A,C)==1
                %set flag not to put in array,
                flag = 1;
                %and break from the loop
                break;
            end
           
        end
        %if the LS was not found in the array already put it into the array
        if flag == 0;
            B (m+1,:,:) = A;
        end
    end
end

%get final size of B array
[m,n,o] = size(B);

%write all LS in array B into a file
for i=1:m
    fprintf(fid2,'%u %u %u %u %u ', B(i,:,:));
    fprintf(fid2,'\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dont read in all values into an array, read in value, check if its in an array, and put it into
%the array if its not