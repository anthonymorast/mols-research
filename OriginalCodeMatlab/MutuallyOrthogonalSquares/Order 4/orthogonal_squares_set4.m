%This script finds sets of 3 squares that are mutually othogonal
%It will only use the squares that belonged to a set of two. 

fid = fopen('triplesO4.dat');

triples = zeros(1,3, 0);

while ~feof(fid)
   a = fscanf(fid, '%u %u',[1 3]);
   triples = cat(3, triples, a);
end

fclose(fid);

%get squares used in the pairs
indexes = getIndexes(triples);
[q,w,e] = size(triples);
[r] = size(indexes);
quads = zeros (1,4,0);
first = 1;

fid = fopen('reduced_order4.dat');
reduced_squares = zeros(4,4,0);

while ~feof(fid)
    A = fscanf(fid, '%u', [4 4]);
    reduced_squares = cat(3, reduced_squares, A);
end

for i=1:e
    currentSquareOne = reduced_squares (:, :, triples(1,1,i));
    currentSquareTwo = reduced_squares (:, :, triples(1,2,i));
    currentSquareThree = reduced_squares (:, :, triples (1,3,i));
    
    for j=1:r
        checkSquare = reduced_squares(:, :, indexes(j));
        contains = 0;
        [p,o,u] = size(quads);
        if (check_latin_square(currentSquareOne, checkSquare) && ...
            check_latin_square(currentSquareTwo, checkSquare) && ...
            check_latin_square(currentSquareThree, checkSquare) )
        
            quad = [triples(1,1,i) triples(1,2,i) triples(1,3,i) indexes(j)]; 
            
            if (first)
                first = 0;
                quads = cat (3, quads, quad);
            end
            for k=1:u
               %if == 3 all same squares
               [c,v] = size(intersect(quads(:,:,k), quad));
               if (v == 3)
                   contains = 1;
                  break; 
               end
            end
            if (~contains)
                quads = cat (3, quads, quad);
            end
        end
    end
end

fid3 = fopen ('quadsO4.dat','w');

[p,o,u] = size(quads);
for i=1:u
    quad = quads(:,:,i);
    contains = 0;
    for j=i+1:u
        if (isequal(quad, quads(:,:,j)))
            contains = 1;
            break;
        end
    end
    if (~contains)
       disp('here');
       fprintf(fid3, '%u %u %u %u\n', quad); 
    end
end

fclose('all');