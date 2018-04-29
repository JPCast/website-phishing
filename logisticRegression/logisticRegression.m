%% The present module aims to classify the data using logistic regression, non regularized, using gradient descent
%% Initialization
clear ; close all; clc

%% ============= Load and prepare data ================
data = load('PhishingData.txt');
c1 = data(:, 1);
c2 = data(:, 2);
c3 = data(:, 3);
c4 = data(:, 4);
c5 = data(:, 5);
c6 = data(:, 6);
c7 = data(:, 7);
c8 = data(:, 8);
c9 = data(:, 9);

X = [c1 c2 c3 c4 c5 c6 c7 c8 c9];
y = data(:, 10);
y = y + 2; % Map labels to interval 1 to 3

m = size(X,1);

%Divide data into training and test
train_x = X([1:946],:); %First 946 rows for training (aproximatly 70% of the dataset)
train_y = y([1:946],:);
test_x = X([947:1353],:); %Following rows for testing
test_y = y([947:1353],:);

%% ============ Compute Cost and Gradient ============

[theta, J_history] = oneVsAll(train_x, train_y, 3, 0.1, 0); % Get thetas for all 
                                              % classifiers
                                              % argument 0 means training
                                              % without regularization



%% ========== Applying regularization ===========

[theta_reg, J_history_reg] = oneVsAll(train_x, train_y, 3, 0.1, 1); % Get thetas for all 
                                              % classifiers
                                              % argument 1 means training
                                              % with regularization                                            
                                              
%% ========== Check hypothesis performance ===========
clc; %remove previous prints
pred_train = predictOneVsAll(theta, train_x);
pred_test = predictOneVsAll(theta, test_x);

fprintf('\nTraining Set Accuracy on training data: %f', mean(pred_train (:) == train_y(:)) * 100);
fprintf('\nTraining Set Accuracy on test data: %f\n', mean(pred_test (:) == test_y(:)) * 100);

%% ========== Plot Cost history ===============
%J_history = real(J_history);
%hold on; % keep previous plot visible
%figure
%plot(J_history);
%title('Evolution of the cost for unregularized logistic regression');
%xlabel('Iteration');
%ylabel('Cost');

%% ========== Check hypothesis performance with regularization ===========
pred_train = predictOneVsAll(theta_reg, train_x);
pred_test = predictOneVsAll(theta_reg, test_x);

fprintf('\nTraining Set Accuracy on training data with regularization: %f', mean(pred_train (:) == train_y(:)) * 100);
fprintf('\nTraining Set Accuracy on test data with regularization: %f\n', mean(pred_test (:) == test_y(:)) * 100);

%% ========== Plot Cost history ===============
%figure
%J_history_reg = real(J_history_reg);
%plot(J_history_reg);
%title('Evolution of the cost for regularized logistic regression');
%xlabel('Iteration');
%ylabel('Cost');


