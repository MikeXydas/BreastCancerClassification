%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\Users\Miifoo\Desktop\Sxolh\anagnorisi\BreastCancerData.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2019/06/13 19:56:36

%% Initialize variables.
filename = 'C:\Users\Miifoo\Desktop\Sxolh\anagnorisi\BreastCancerData.csv';
delimiter = ',';

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
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
 
%% KNN model training 
knnModel = fitcknn(normalisedData, labels, 'NumNeighbors',5);

%%Naive Bayes model training
naiveB = fitcnb(normalisedData, labels);

%%SVM model training
svmModel = fitcsvm(normalisedData, labels);

%%Decision tree model training
dTree = fitctree(normalisedData, labels);



cvmdl = crossval(dTree);
kfoldLoss(cvmdl)