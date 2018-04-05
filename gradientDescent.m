function [theta] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples

for iter = 1:num_iters
        
    [j, gradJ] = costFunction(theta, X, y);
    theta=theta-alpha*gradJ';
end
end



