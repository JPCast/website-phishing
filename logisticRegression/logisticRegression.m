%% The present module aims to classify the data using logistic regression, non regularized, using gradient descent
%% Initialization
clear ; close all; clc

%% ============= Load and prepare data ================
load('training.mat');
load('labels.mat');

X = X+2; %Map values of features to integer interval 1 to 3
y = y + 2; % Map labels to interval 1 to 3

m = size(X,1);
training_length = round(0.65*m);
crossValidation_length = round(0.2*m);

%Divide data into training, cross validation, and testing sets
train_x = X([1:training_length],:); %First 946 rows for training (aproximatly 70% of the dataset)
train_y = y([1:training_length],:);

cv_x = X([training_length+1:training_length+crossValidation_length],:); %Next rows for cross validation
cv_y = y([training_length+1:training_length+crossValidation_length],:);

test_x = X([training_length+crossValidation_length+1:m],:); %Last rows for testing
test_y = y([training_length+crossValidation_length+1:m],:);


%% ============ Compute Cost and Thetas ============

[theta, J_history] = oneVsAll(train_x, train_y, 3, 0.1, 0); % Get thetas for all 
                                              % classifiers
                                              % argument 0 means training
                                              % without regularization



%% ========== Compute Cost and Thetas applying regularization ===========

[theta_reg, J_history_reg] = oneVsAll(train_x, train_y, 3, 0.1, 1); % Get 
                                              %thetas for all classifiers
                                              % argument 1 means training
                                              % with regularization                                            
                                
                                              
%% ========== Check hypothesis performance without regularization ===========
clc; %remove previous prints
pred_train = predictOneVsAll(theta, train_x);
pred_test = predictOneVsAll(theta, cv_x);

fprintf('\nTraining Set Accuracy on training data: %f', mean(pred_train (:) == train_y(:)) * 100);
fprintf('\nTraining Set Accuracy on test data: %f\n', mean(pred_test (:) == cv_y(:)) * 100);

% Generate confusion matrix for test set
pred_test = pred_test';
confusion_matrix = confusionmat(pred_test, cv_y); 

% Determine performance metrics of the model on the test data
%Acuracy
accuracy = (confusion_matrix(1,1)+confusion_matrix(2,2)+confusion_matrix(3,3))/length(cv_y)*100;

%Recall
recall_legit = confusion_matrix(3,3)/(confusion_matrix(1,3)+confusion_matrix(2,3)+confusion_matrix(3,3));
recall_suspicious = confusion_matrix(2,2)/(confusion_matrix(1,2)+confusion_matrix(2,2)+confusion_matrix(3,2));
recall_phishing = confusion_matrix(1,1)/(confusion_matrix(1,1)+confusion_matrix(2,1)+confusion_matrix(3,1));

%Precision
precision_legit = confusion_matrix(3,3)/(confusion_matrix(3,1)+confusion_matrix(3,2)+confusion_matrix(3,3));
precision_suspicious = confusion_matrix(2,2)/(confusion_matrix(2,1)+confusion_matrix(2,2)+confusion_matrix(2,3));
precision_phishing = confusion_matrix(1,1)/(confusion_matrix(1,1)+confusion_matrix(1,2)+confusion_matrix(1,3));

fprintf('\nConfusion matrix');
confusion_matrix

fprintf('\nAccuracy = %f', accuracy);
fprintf('\nRecall (legit) = %f', recall_legit);
fprintf('\nRecall (suspicious) = %f', recall_suspicious);
fprintf('\nRecall (phishing) = %f', recall_phishing);
fprintf('\nPrecision (legit) = %f', precision_legit);
fprintf('\nPrecision (suspicious) = %f', precision_suspicious);
fprintf('\nPrecision (phishing) = %f\n', precision_phishing);



%% ========== Check hypothesis performance with regularization ===========
pred_train = predictOneVsAll(theta_reg, train_x);
pred_test = predictOneVsAll(theta_reg, cv_x);

fprintf('\nTraining Set Accuracy on training data with regularization: %f', mean(pred_train (:) == train_y(:)) * 100);
fprintf('\nTraining Set Accuracy on test data with regularization: %f\n', mean(pred_test (:) == cv_y(:)) * 100);

% Generate confusion matrix for test set
pred_test = pred_test';
confusion_matrix = confusionmat(pred_test, cv_y); 

fprintf('\nConfusion matrix');
confusion_matrix



%% ========== Check regularization performance for several values of lambda ===============
% 
% lambda = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
% results_train = zeros(length(lambda));
% results_test = zeros(length(lambda));
% 
% for i = 1:length(lambda)
%     [theta_reg, J_history_reg] = oneVsAll(train_x, train_y, 3, lambda(i), 1); % Get 
%                                               %thetas for all classifiers
%                                               % argument 1 means training
%                                               % with regularization   
%                                               pred_train = predictOneVsAll(theta_reg, train_x);
%                                               
%     pred_train = predictOneVsAll(theta_reg, train_x);
%     pred_test = predictOneVsAll(theta_reg, test_x);
%     
% 
%     results_train(i) = (mean(pred_train (:) == train_y(:)) * 100);
%     results_test(i) = (mean(pred_test (:) == test_y(:)) * 100);
% end
% clc;
% figure
% plot(results_train);
% figure
% plot(results_test);


%% ========== Adding polynomial features to solve high bias (underfit) ================
% code from https://stackoverflow.com/questions/33660799/feature-mapping-using-multi-variable-polynomial#33886052
n_vars = 9;     % number of variables
max_degree  = 3;     % order of polynomial
stacked = zeros(0, n_vars); %this will collect all the coefficients...    
for d = 1:max_degree          % for degree 1 polynomial to degree 'order'
    stacked = [stacked; mapFeature(n_vars, d)];
end
newX = zeros(m, size(stacked,1));
for(i = 1:size(stacked,1))
    accumulator = ones(m, 1);
    for(j = 1:n_vars)
        accumulator = accumulator .* X(:,j).^stacked(i,j);
    end
    newX(:,i) = accumulator;
end

m = size(X,1);
training_length = round(0.65*m);
crossValidation_length = round(0.2*m);

%Divide data into training, cross validation, and testing sets
train_x = newX([1:training_length],:); %First 946 rows for training (aproximatly 70% of the dataset)
train_y = y([1:training_length],:);

cv_x = newX([training_length+1:training_length+crossValidation_length],:); %Next rows for cross validation
cv_y = y([training_length+1:training_length+crossValidation_length],:);

test_x = newX([training_length+crossValidation_length+1:m],:); %Last rows for testing
test_y = y([training_length+crossValidation_length+1:m],:);

[theta, J_history] = oneVsAll(train_x, train_y, 3, 0.1, 0); % Get thetas for all 
                                              % classifiers
                                              % argument 0 means training
                                              % without regularization

pred_train = predictOneVsAll(theta, train_x);
pred_test = predictOneVsAll(theta, cv_x);

fprintf('\nTraining Set Accuracy on training data with polymial features of n = %f: %f', max_degree, mean(pred_train (:) == train_y(:)) * 100);
fprintf('\nTraining Set Accuracy on test data with polynomial features of n = %f: %f\n', max_degree, mean(pred_test (:) == cv_y(:)) * 100);

% Generate confusion matrix for test set
pred_test = pred_test';
confusion_matrix = confusionmat(pred_test, cv_y); 

fprintf('\nConfusion matrix');
confusion_matrix

%% ================Adding regularization to the polynomial features ========
[theta_reg, J_history_reg] = oneVsAll(train_x, train_y, 3, 0.01, 1); % Get thetas for all 
                                              % classifiers
                                              % argument 1 means training
                                              % with regularization
                                              
pred_train = predictOneVsAll(theta_reg, train_x);
pred_test = predictOneVsAll(theta_reg, cv_x);

fprintf('\nTraining Set Accuracy on training data with polymial features of n = %f and regularization: %f', max_degree, mean(pred_train (:) == train_y(:)) * 100);
fprintf('\nTraining Set Accuracy on test data with polynomial features of n = %f and regularization: %f\n', max_degree, mean(pred_test (:) == cv_y(:)) * 100);

% Generate confusion matrix for test set
pred_test = pred_test';
confusion_matrix = confusionmat(pred_test, cv_y); 

fprintf('\nConfusion matrix');
confusion_matrix
