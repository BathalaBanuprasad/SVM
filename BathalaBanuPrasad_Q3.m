clear all
close all

%% Generate data
X = randn(500,2);
Y = double((X(:,1) > X(:,2)))*2-1;
X = zscore(X);
sigma = 1;
X_test = X(201:end,:);
X = X(1:200,:);
Y_test = Y(201:end);
Y = Y(1:200);

X = X';
y = Y';
s = size(y);
n = s(2);

ERR = [];
%% Model parameters
sigma = [1e-2,1e-1,0.5,1e1,1e2];
for sig = sigma
    [~,~,acc,~,~] = BathalaBanuPrasad_SVM_Gau(X,y,sig);
    ERR = [ERR, 1-acc];
end

%Plot error
figure
semilogx(sigma,ERR)
hold on
scatter(sigma,ERR,'filled')
hold off
xlabel('Value of sigma')
ylabel('Error');
title('Plot of Error vs sigma for Gaussian kernel')
