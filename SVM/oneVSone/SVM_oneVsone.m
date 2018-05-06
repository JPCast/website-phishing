data=csvread("PhishingData.txt");
sizeData=size(data);
trainingDataPer=0.6;
sizeOfTrainningData=round(sizeData(1,1)*trainingDataPer);

%trainingData=data(1:sizeOfTrainningData,:);
%define after training data
trainingData=[];
testData=data(sizeOfTrainningData+1:sizeData(1,1),:);

%model = svmtrain(trainingData(:,10), trainingData(:,[1,9]));
%pred =  svmpredict(testData(:,10), testData(:,[1,9]), model);

labels=unique(data(:,10))

labelCombinations = nchoosek(labels,2);            %# 1-vs-1 combination of labels 
models = cell(size(labelCombinations,1),1);            %# store binary-classifers
% predTest = cell(1,length(models)); 
predTest = zeros(sizeData(1,1)-sizeOfTrainningData,length(models));

for m=1:length(models)
    trainingData=[];
    %testData=[];
    for label=1:size(labelCombinations,2)
        trainingData=[trainingData; data(data(1:sizeOfTrainningData,10)==labelCombinations(m,label),:)];
   %     testData=[testData; data(data(sizeOfTrainningData+1:sizeData(1,1),10)==labelCombinations(m,label),:)];
    end
    models{m} = svmtrain(trainingData(:,10),trainingData(:,[1:9]));
    predTest(:,m) = svmpredict(testData(:,10), testData(:,[1:9]), models{m})
   % svmclassify(models{m},testData(:,[1:9]));
end
pred = mode(predTest,2);   %# clasify as the class receiving most votes

cmat = confusionmat(testData(:,10),pred);
acc = 100*sum(diag(cmat))./sum(cmat(:));
fprintf('SVM (1-against-1):\naccuracy = %.2f%%\n', acc);
fprintf('Confusion Matrix:\n'), disp(cmat)

%get index of each class
%indexClass0=data(:,10)==0;
%indexClass1=data(:,10)==1;
%indexClass11=data(:,10)==-1;


%testeData= [data(indexClass0,:); data(indexClass1,:);data(indexClass11,:) ];

%model = fitcsvm(testeData(:,1:9),testeData(:,10));

%model = svmtrain(testeData(:,10), testeData(:,[1,9]));
%pred =  svmpredict(testeData(:,10), testeData(:,[1,9]), model);

%model = svmtrain(testeData(:,10),testeData(:,[1,9]));
%predictions = svmpredict(1,data(indexClass0,[1:9]),data(indexClass0,10),model);
