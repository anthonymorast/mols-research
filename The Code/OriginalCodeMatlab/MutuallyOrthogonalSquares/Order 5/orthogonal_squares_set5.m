%This script finds sets of 3 squares that are mutually othogonal
%It will only use the squares that belonged to a set of two. 

fid = fopen('quadsO5.dat');

quads = zeros(1,4, 0);

while ~feof(fid)
   a = fscanf(fid, '%u %u',[1 4]);
   quads = cat(3, quads, a);
end

fclose(fid);

%get squares used in the pairs
indexes = getIndexes(quads);
[q,w,e] = size(quads);
[r] = size(indexes);
quints = zeros (1,5,0);
first = 1;

fid = fopen('reduced_squares_o5.dat');
reduced_squares = zeros(5,5,0);

while ~feof(fid)
    A = fscanf(fid, '%u', [5 5]);
    reduced_squares = cat(3, reduced_squares, A);
end

for i=1:e
    currentSquareOne = reduced_squares (:, :, quads(1,1,i));
    currentSquareTwo = reduced_squares (:, :, quads(1,2,i));
    currentSquareThree = reduced_squares (:, :, quads(1,3,i));
    currentSquareFour = reduced_squares (:, :, quads(1,4,i));
    
    for j=1:r
        checkSquare = reduced_squares(:, :, indexes(j));
        contains = 0;
        [p,o,u] = size(quints);
        if (check_latin_square(currentSquareOne, checkSquare) && ...
            check_latin_square(currentSquareTwo, checkSquare) && ...
            check_latin_square(currentSquareThree, checkSquare) && ...
            check_latin_square(currentSquareFour, checkSquare))
        
            quint = [triples(1,1,i) triples(1,2,i) triples(1,3,i) indexes(j)]; 
            
            if (first)
                first = 0;
                quints = cat (3, quints, quint);
            end
            for k=1:u
               %if == 3 all same squares
               [c,v] = size(intersect(quints(:,:,k), quint));
               if (v == 5)
                   contains = 1;
                  break; 
               end
            end
            if (~contains)
                quints = cat (3, quints, quint);
            end
        end
    end
end

fid3 = fopen ('quintsO4.dat','w');

[p,o,u] = size(quints);
for i=1:u
    quints = quints(:,:,i);
    contains = 0;
    for j=i+1:u
        if (isequal(quint, quints(:,:,j)))
            contains = 1;
            break;
        end
    end
    if (~contains)
       fprintf(fid3, '%u %u %u %u\n', quint); 
    end
end

fclose('all');