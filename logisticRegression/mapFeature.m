function result = mapFeature(n_numbers, d)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%
    if(n_numbers<=1)
        result = d;
    else
        result = zeros(0, n_numbers);    
        for(i = d:-1:0)
            rc = mapFeature(n_numbers - 1, d - i);
            result = [result; i * ones(size(rc,1), 1), rc];
        end    
    end
end