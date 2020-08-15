 function [W,b,acc,X_sv,y_sv] = BathalaBanuPrasad_SVM(X,y,itr)
    %% Both X and y has samples along coloumns and features along rows
    %% itr = 0, to ignore maximum number of iterations
    n = size(y,2);
    W = zeros(size(X,1),1);
    b = 0;
    %% Solve the optimisation problem
    Y = diag(y);
    sigma = X'*X; %We need samples along col and features along rows
    
    %Update maximum number of iterations for every iteration
    if itr
        cvx_solver_settings( 'maxit', itr)
    end
    N = size(X,2);

    cvx_begin
        cvx_precision best
        variable a_m(N);
        maximise (ones(1,N)*(a_m)-0.5*quad_form(y'.*a_m,sigma));
        subject to
            a_m >= 0;
            y*(a_m) == 0;
    cvx_end
    %% Obtain support vectors (SV)
    l = a_m;
    l(l<1e-5) = 0;
    indices = (l>0)';

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
        b = b+(y_sv(i)-W'*X_sv(:,i));
    end
    b = b/N;

    %% Accuracy computation
    predict = sign(W'*X-b);
    acc = mean(predict==y);
end