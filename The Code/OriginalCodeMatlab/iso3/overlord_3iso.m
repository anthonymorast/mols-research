function overlord_3iso ( root )
%
%
%accepts input root, which is the root name of the file to be writte. this
%program reads in one latin square at a time, increments each index,
%permutes the symbols, and writes them each to an idividual file. root is
%used to appaned the numbers 1-56 on these files. 
%

fid = fopen ('mckay_nls5.txt');

for i=1:56
    A = fscanf(fid, '%u',[5 5]);
    A = A';
    for j=1:5
        for k=1:5
            A(j,k) = A(j,k) + 1;
        end
    end
    filename = root;
    filename = strcat(filename,num2str(i),'.dat');
    filename2 = filename;
    permu_iso_2(A, filename);
    eq_ls2(filename2);
    
end

fclose('all');