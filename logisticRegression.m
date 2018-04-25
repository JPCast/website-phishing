%% The present module aims to classify the data using logistic regression, non regularized, using gradient descent
%% Initialization
clear ; close all; clc

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

%% ============ Part 2: Compute Cost and Gradient ============

[theta] = oneVsAll(train_x, train_y, 3, 0.1); % Get thetas for all 
                                              % classifiers

%% ========== Part 3: Check hypothesis performance ===========
pred = predictOneVsAll(theta, test_x);

fprintf('\nTraining Set Accuracy: %f\n', mean(pred (:) == test_y(:)) * 100);


