%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 9;  % 9 features
hidden_layer_size = 6;   
num_labels = 3;          % 3 labels, from 1 to 3  (-1 is mapped to label 1)

%% =========== Part 1: Load and Visualize Some Images =============
load 'training.mat' ; % Load Training Data
m = size(X, 1); %number of examples

%% =========== Part 2: Determine NN Theta Parameters ===============





%% =========== Part 3: Implement Predict =================
%  After training the NN, we would like to use it to predict the labels. 
%You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', ?????);

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(?,?,?);
     
    fprintf('\n NN Prediction: %d (True label %d) \n', ?, ? )
    
     % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
end
