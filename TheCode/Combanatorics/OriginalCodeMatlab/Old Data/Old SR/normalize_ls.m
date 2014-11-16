function C = normalize_ls(A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function takes any latin square and swaps
% rows and columns until it is normalized.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m,n] = size(A);

%normalize the latin square produced by the permutation
%swap columns
if( A(1,1) ~= 1)
   %find column starting with 1
   for i=1:n
       if A(1,i) == 1
           swap = i;
           break
       end
   end
   %swap columns
   D = A(:,1);
   A(:,1) = A(:,swap);
   A(:,swap) = D;
end

if (A(1,2) ~= 2)
    %find column starting with 2
    for i=1:m
        if A(1,i) == 2
            swap = i;
            break
        end
    end
    D = A(:,2);
    A(:,2) = A(:,swap);
    A(:,swap) = D;
end

if (A(1,3) ~= 3)
    %find column starting with 3
    for i=1:m
        if A(1,i) == 3
            swap = i;
            break
        end
    end
    D = A(:,3);
    A(:,3) = A(:,swap);
    A(:,swap) = D;
end

%swap rows

if (A(2,1) ~= 2)
    %find row starting with 2
    for i=1:m
        if A(i,1) == 2
            swap = i;
            break
        end
    end
    D = A(2,:);
    A(2,:) = A(swap,:);
    A(swap,:) = D;
end

if (A(3,1) ~= 3)
    %find row starting with 3
    for i=1:m
        if A(i,1) == 3
            swap = i;
            break
        end
    end
    D = A(3,:);
    A(3,:) = A(swap,:);
    A(swap,:) = D;
end

C = A;