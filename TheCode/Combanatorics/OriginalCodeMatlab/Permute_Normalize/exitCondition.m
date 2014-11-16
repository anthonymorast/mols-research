function b = exitCondition( b, c, n )
%EXITCONDITION
%   Checks if we are ready to exit the main loop in move_to_normal
    [h,i,l] = size(c);

    for i=1:l
        C = c(:,:,i);
        [h,i,p] = size(b);
        for j=1:p
           if isequal(b(:,:,j),C)
               b(:,:,j) = [];
               break;
           end
        end
    end
end
