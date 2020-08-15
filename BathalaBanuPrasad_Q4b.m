clear all
close all

%% Prepare training and test data set
load('data.mat');
load('label.mat');

imageTrain = reshape(imageTrain,784, 5000)'/255;
imageTest = reshape(imageTest,784, 500)'/255;

class1 = 1;
class2 = 7;
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

%% Run SVM algorithms
%Hard SVM
[W,b,acc,X_sv,y_sv] = BathalaBanuPrasad_SVM(X_train', y_train',100);
acc_hard = accuracy(W,b,X_test', y_test')

%SVM with gaussian kernel
[W_g,b_g,acc_g,X_sv_g,y_sv_g] = BathalaBanuPrasad_SVM_Gau(X_train', y_train',0.5);
acc_gau = accuracy(W_g,b_g,X_test', y_test')

[W_g1,b_g1,acc_g1,X_sv_g1,y_sv_g1] = BathalaBanuPrasad_SVM_Gau(X_train', y_train',0.1);
acc_gau_2 = accuracy(W_g1,b_g1,X_test', y_test')
