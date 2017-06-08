function indexes = getIndexes(pairs)

%returns a list of the new indexes to be checked
[h,k,l] = size(pairs);
indexes = zeros(0);
first = 1;

for i = 1:l
    containsOne = 0;
    containsTwo = 0;
    containsThree = 0;
    
    one = pairs(1,1,i);
    two = pairs(1,2,i);
    three = pairs (1,3,i);
    
    if (first)
       first = 0;
       indexes = cat(1, indexes, one);
       indexes = cat(1, indexes, two);
       indexes = cat(1, indexes, three);
       loop = 2;
    end
    if (~first)
        for j=1:loop
            if indexes(j) == one
                containsOne = 1;
            end
            if indexes(j) == two
                containsTwo = 1;
            end
            if indexes(j) == three
                containsThree = 1;
            end
        end
    end
    
    if (~containsOne)
        indexes = cat (1, indexes, one);
        loop = loop + 1;
    end
    if (~containsTwo)
        indexes = cat (1, indexes, two);
        loop = loop + 1;
    end
    if (~containsThree)
        indexes = cat (1, indexes, three);
        loop = loop + 1;
    end
end