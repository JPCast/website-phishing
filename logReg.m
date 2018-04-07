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
y = y + 2; % map labels to interval 1 to 3

%Divide data into training and test
train_x = X([1:946],:); %First 946 rows for training (aproximatly 70% of the dataset)
train_y = y([1:946],:);
test_x = X([947:1353],:); %Following rows for testing
test_y = y([947:1353],:);

%% ============ Part 1: Visualize the data ===================



%% ============ Part 2: Compute Cost and Gradient ============

[m, n] = size(train_x); % m training examples, n features

% Add extra column of 1 to X
%column_ones = ones(m, 1);
%train_x = [column_ones train_x];

% Initialize fitting parameters =0
%theta = zeros(n+1, 1); % theta is a column vector with the number of features plus 1
%iterations = 1500;
%alpha = 0.01;

% run gradient descent
%[theta] = gradientDescent(train_x, train_y, theta, alpha, iterations);

% print theta to screen
%fprintf('Theta found by gradient descent: ');
%fprintf('%f \n', theta);


[theta] = oneVsAll(train_x, train_y, 3, 0.1); % Get thetas for all 
                                            % classifiers



%% ========== Part 3: Check hypothesis performance ===========
pred = predictOneVsAll(theta, test_x);

fprintf('\nTraining Set Accuracy: %f\n', mean(pred (:) == test_y(:)) * 100);


