%This script finds sets of 3 squares that are mutually othogonal
%It will only use the squares that belonged to a set of two. 

fid = fopen('mutually_orthogonal_squaresO4.dat');

pairs = zeros(1,2, 0);

while ~feof(fid)
   a = fscanf(fid, '%u %u',[1 2]);
   pairs = cat(3, pairs, a);
end

fclose(fid);

%get squares used in the pairs
indexes = getIndexes(pairs);
[q,w,e] = size(pairs);
[r] = size(indexes);
triples = zeros (1,3,0);
first = 1;

fid = fopen('reduced_squares_order4.dat');
reduced_squares = zeros(4,4,0);

while ~feof(fid)
    A = fscanf(fid, '%u', [4 4]);
    reduced_squares = cat(3, reduced_squares, A);
end


for i=1:e
    currentSquareOne = reduced_squares (:, :, pairs(1,1,i));
    currentSquareTwo = reduced_squares (:, :, pairs(1,2,i));
    for j=1:r
        checkSquare = reduced_squares(:, :, indexes(j));
        contains = 0;
        [p,o,u] = size(triples);
        if (check_latin_square(currentSquareOne, checkSquare) && ...
            check_latin_square(currentSquareTwo, checkSquare))
            triple = [pairs(1,1,i) pairs(1,2,i) indexes(j)]; 
            if (first)
                first = 0;
                triples = cat (3, triples, triple);
            end
            for k=1:u
               %if == 3 all same squares
               [c,v] = size(intersect(triples(:,:,k), triple));
               if (v == 3)
                   contains = 1;
                  break; 
               end
            end
            if (~contains)
                triples = cat (3, triples, triple);
            end
        end
    end
end

fid3 = fopen ('triplesO4.dat','w');

[p,o,u] = size(triples);
for i=1:u
    triple = triples(:,:,i);
    contains = 0;
    for j=i+1:u
        if (isequal(triple, triples(:,:,j)))
            contains = 1;
            break;
        end
    end
    if (~contains)
        disp('here');
       fprintf(fid3, '%u %u %u \n', triple); 
    end
end

fclose('all');