%function p32()
close all
clear all
%% Get Data and plot the data
% With 50 positive and negative examples
% [X,y] = svm_gendata(50,50);
% y(X>0) = 1;
% y(X<0) = -1;
%% Create Training data:
X = randn(500,2);
% Y = double((X(:,1)+1.0*randn(1000,1)>X(:,2)))*2-1;
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

ACC = [];
max_itr = 25;

%% Iterating over multiple number of maximum number of iterations
for itr = 1:max_itr
    fprintf('Iteration: %d/%d',itr,max_itr)
    [W,b,acc,X_sv,y_sv] = BathalaBanuPrasad_SVM(X,y,itr);
    ACC = [ACC, acc];
end

% Clear cvxr settings
cvx_solver_settings -clear

%% Accuracy plot
figure
hold on
plot(1-ACC)
%plot(ACC)
hold off
%legend('Error','Accuracy')
xlabel('Number of iterations')
ylabel('Error')%/Accuracy')
title('Error vs Number of iterations plot')

%% Visualise the data

% SV of class 1 (y>0)
sv1 = X_sv(:,y_sv>0);

% SV of class 2 (y<0)
sv2 = X_sv(:,y_sv<0);

class1 = X(:, y>0);
class2 = X(:, y<0);
% Display data
figure
scatter(class1(1,:),class1(2,:),[],'r','filled')
hold on
scatter(class2(1,:),class2(2,:),[],'b','filled')

% display support vectors
scatter(sv1(1,:),sv1(2,:),[],'y','filled')
scatter(sv2(1,:),sv2(2,:),[],'g','filled')

ax_min = min(X');
ax_max = max(X');

ax = ax_min(1):0.1:ax_max(1);
ay = ax_min(2):0.1:ax_max(2);
[ax,ay] = meshgrid(ax,ay);
Z = zeros(size(ax));
ax_s = size(ax);

for i = 1:ax_s(1)
    for j = 1:ax_s(2)
        Z(i,j) = W'*[ax(i,j);ay(i,j)]+b;
    end
end
contour(ax,ay,Z,[-1,-1], 'LineWidth',1)
contour(ax,ay,Z,[1,1], 'LineWidth',1)
contour(ax,ay,Z,[0,0], 'LineWidth',1)
%contout(ax,ay,Z,'ShowText','on')
legend('Class 1','Class -1','SV of Class 1','SV of Class -1', 'sep -1', 'sep 1', 'sep 0')

