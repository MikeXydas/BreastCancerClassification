%% Initialize variables - care this will have to change.
filename = 'BreastCancerData.csv';
delimiter = ',';

%% Format string for each line of text:
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);


%% Allocate imported array to column variable names
VarName1 = dataArray{:, 1};
VarName2 = dataArray{:, 2};
VarName3 = dataArray{:, 3};
VarName4 = dataArray{:, 4};
VarName5 = dataArray{:, 5};
VarName6 = dataArray{:, 6};
VarName7 = dataArray{:, 7};
VarName8 = dataArray{:, 8};
VarName9 = dataArray{:, 9};
VarName10 = dataArray{:, 10};
VarName11 = dataArray{:, 11};


%% Clear temporary variables
clearvars filename delimiter formatSpec fileID ans dataArray;

%% Create the nx11 matrix

dataSet = [VarName1, VarName2, VarName3, VarName4, VarName5, VarName6, VarName7, VarName8, VarName9, VarName10, VarName11]; 
dataSet = double(dataSet);
labels = dataSet(:, 11);

%% Normalise values in [0,1]

normalisedData = [];

for k=2:size(dataSet, 2) - 1
    normalisedData = [normalisedData, (dataSet(:, k) - min(dataSet(:, k))) / (max(dataSet(:, k)) - min(dataSet(:, k)))];
end
 

% for i = 1:10
%     [train,test] = crossvalind('Kfold',labels,10);
%     mdl = fitcknn(normalisedData(train,:),labels(train),'NumNeighbors',3);
%     predictions = predict(mdl,normalisedData(test,:));
%     classperf(cp,predictions,test);
% end

indices = crossvalind('Kfold',labels,10);




%% KNN model training 
cpKnn = classperf(labels);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcknn(normalisedData(train,:),labels(train),'NumNeighbors',5);
    predictions = predict(mdl,normalisedData(test,:));
    classperf(cpKnn,predictions,test);
end

disp('KNN:');
fprintf('Accuracy: %f\n', 1 - cpKnn.ErrorRate);
fprintf('Sensitivity: %f\n', cpKnn.Sensitivity);
fprintf('Specificity: %f\n', cpKnn.Specificity);

fprintf('\n');

%% Naive Bayes model training
cpNBayes = classperf(labels);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcnb(normalisedData(train,:),labels(train));
    predictions = predict(mdl,normalisedData(test,:));
    classperf(cpNBayes,predictions,test);
end

disp('Naive Bayes:');
fprintf('Accuracy: %f\n', 1 - cpNBayes.ErrorRate);
fprintf('Sensitivity: %f\n', cpNBayes.Sensitivity);
fprintf('Specificity: %f\n', cpNBayes.Specificity);

fprintf('\n');

%% SVM model training
cpSvm = classperf(labels);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitcsvm(normalisedData(train,:),labels(train));
    predictions = predict(mdl,normalisedData(test,:));
    classperf(cpSvm,predictions,test);
end

disp('SVM:');
fprintf('Accuracy: %f\n', 1 - cpSvm.ErrorRate);
fprintf('Sensitivity: %f\n', cpSvm.Sensitivity);
fprintf('Specificity: %f\n', cpSvm.Specificity);

fprintf('\n');

%% Decision tree model training
cpDTree = classperf(labels);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdl = fitctree(normalisedData(train,:),labels(train));
    predictions = predict(mdl,normalisedData(test,:));
    classperf(cpDTree,predictions,test);
end

disp('Decision Tree:');
fprintf('Accuracy: %f\n', 1 - cpDTree.ErrorRate);
fprintf('Sensitivity: %f\n', cpDTree.Sensitivity);
fprintf('Specificity: %f\n', cpDTree.Specificity);

fprintf('\n');

%%dTree


%%cvmdl = crossval(dTree);
%%size(cvmdl.Y);
%%disp(all(cvmdl.Y==labels));
%%disp('Accuracy:');

%%disp(1 - kfoldLoss(cvmdl));


