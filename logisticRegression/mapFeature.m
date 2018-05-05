function result = mapFeature(n_numbers, d)
% Feature mapping using multi-variable polynomial
% Code source: https://stackoverflow.com/questions/33660799/feature-mapping-using-multi-variable-polynomial#33886052

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