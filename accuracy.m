function acc = accuracy(W,b,X,y)
%% X, y has data samples along columns and fetaures along rows
%% W is a column vector
    predict = sign(W'*X+b);
    acc = mean(predict==y);
end