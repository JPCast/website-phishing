function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in matrix X. Note that each row of X contains one example (image). 
% all_theta is a matrix where i-th row is a trained logistic
%  regression theta vector for  i-th class. You should get p as a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m,1), X];

z = X *all_theta';

h = (1./ (1 + exp(-z) ));
[var,ind]=max(h');
p=ind;

end