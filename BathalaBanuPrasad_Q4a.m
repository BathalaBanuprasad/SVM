clear all
close all

load('data.mat');
load('label.mat');

imageTrain = reshape(imageTrain,784, 5000)'/255;
imageTest = reshape(imageTest,784, 500)'/255;

class1 = 6;
class2 = 8;

% Training data
id_1 = find(labelTrain == class1);
id_2 = find(labelTrain == class2);
X_train = imageTrain([id_1; id_2], :);
y_train = labelTrain([id_1; id_2]);

% Test data
idTest_1 = find(labelTest == class1);
idTest_2 = find(labelTest == class2);
X_test = imageTest([idTest_1; idTest_2], :);
y_test = labelTest([idTest_1; idTest_2]);

y_train(y_train == class1) = -1;
y_train(y_train == class2) = 1;

y_test(y_test == class1) = -1;
y_test(y_test == class2) = 1;



%% Prepare parameters for model
n = size(X_train,1);
m = size(X_train,2);
b = 0;
l = randn(size(y_train));
Y = diag(y_train);
sigma = X_train'*X_train; %We need samples along col and features along rows
o = ones(1,n);

cvx_begin
    variable W(m)
    variable b
    minimize(norm(W))
    subject to
        Y'*(W'*X_train'+b)' >= 1
cvx_end

acc_test = accuracy(W,b,X_test',y_test')