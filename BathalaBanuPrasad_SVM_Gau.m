function [W,b,acc,X_sv,y_sv] = BathalaBanuPrasad_SVM_Gau(X,y,sig)
%% Both X and y has samples along coloumns and features along rows
n = size(y,2);
W = zeros(size(X,1),1);
b = 0;
%% Solve the optimisation problem
l = randn(size(y));
Y = diag(y);
K = zeros(n); %We need samples along col and features along rows
for i = 1:n
    for j =1:n
        K(i,j) = -(1/(sig^2))*norm(X(:,i)-X(:,j))^2;
    end
end
K = exp(K);
o = ones(1,n);

cvx_begin
    variable l(n);
    maximise(o*l - 0.5.*quad_form(l,Y*K*Y)) %0.5*quad_form(lambda,Y*sigma*Y))
    subject to
        l >= 0;
        y*l == 0;
cvx_end
%end

%% Obtain support vectors (SV)
l(l<1e-5) = 0;
indices = (l>0)';

% SV of class 1 (y>0)
sv1 = X(:,indices&(y>0));

% SV of class 2 (y<0)
sv2 = X(:,indices&(y<0));


%% Compute weight vector

N = sum(indices);
y_sv = y(indices);
X_sv = X(:,indices);
l = l(l>0);

for i = 1:N
    temp = l(i)*y_sv(i);
    W = W+temp*X_sv(:,i);
end

%% Get bias
for i = 1:N
    b = b+(y_sv(i)-W'*X_sv(:,i));% - W;
end
b = b/N;

%% Accuracy computation
predict = sign(W'*X-b);
acc = mean(predict==y);
end
