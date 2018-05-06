% ======== Initializaton ==============
close all; clear all; clc;
load('training.mat');
load('labels.mat');

% m = size(X,1);
% training_length = round(0.65*m);
% crossValidation_length = round(0.2*m);
% 
% %Divide data into training, cross validation, and testing sets
% train_x = X([1:training_length],:); %First 946 rows for training (aproximatly 70% of the dataset)
% train_y = labels([1:training_length],:);
% 
% cv_x = X([training_length+1:training_length+crossValidation_length],:); %Next rows for cross validation
% cv_y = labels([training_length+1:training_length+crossValidation_length],:);
% 
% test_x = X([training_length+crossValidation_length+1:m],:); %Last rows for testing
% test_y = labels([training_length+crossValidation_length+1:m],:);
% 
% X_training = [train_x; cv_x];
% Y_labels = [train_y; cv_y];
% 
% X_training = X_training';
% Y_labels = Y_labels';
X = X';
labels = labels';



RandStream.setGlobalStream (RandStream ('mrg32k3a','Seed', 1234)); % Use always the same seed


% ======== Build the network ==========
net = patternnet([15, 4]); % 1 hidden layer, i neurons
net.divideParam.trainRatio = 65/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 15/100;

% ======== Train the netowrk ==========
[net,tr] = train(net,X,labels);
nntraintool;

% ======== Check accuracy =============
testX = X(:,tr.testInd);
testT = labels(:,tr.testInd);
testY = net(testX);
testIndices = vec2ind(testY);
[c,cm] = confusion(testT,testY);
fprintf('Accuracy: %f\n', (100*(1-c)));

