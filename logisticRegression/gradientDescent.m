function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters, reg, c, J_history)

m = length(y); % number of training examples

for iter = 1:num_iters
    if(reg==0)
        [j, gradJ] = costFunction(theta, X, y);
    else
        [j, gradJ] = costFunctionReg(theta, X, y, 1);
    end;
    theta=theta-alpha*gradJ';
    J_history(iter, c) = computeCost(X,y,theta);
end
end



