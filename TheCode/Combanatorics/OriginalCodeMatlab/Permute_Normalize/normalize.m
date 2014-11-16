function B = normalize(A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Anthony Morast
% Sr. Research
%
% A program that takes a LS as input and returns the normalized square as
% output. Works for LS of any order. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m,n] = size(A);

%permute rows into 1,2,3,...,n positions
for i=1:m
    %if the current row, first column isn't correct 
    if A(i,1) ~= i     
        %swap row into it's proper position
        temp = A(A(i,1),:);
        A(A(i,1),:) = A(i,:);
        A(i,:) = temp;
    end
end

%permute columns into 1,2,3,...,n positions
for i=1:n
    %if the current column, first row isn't correct 
    if A(i,1) ~= i     
        %swap row into it's proper position
        temp = A(:,A(1,i));
        A(:,A(1,i)) = A(:,i);
        A(:,i) = temp;
    end
end

B = A;