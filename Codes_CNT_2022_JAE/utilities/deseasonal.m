% DESEASONAL.M
% This program fits monthly seasonal dummies to a univariate time series 
% and returns the residuals

function [ehat,yhat]=deseasonal(y);

t=length(y);

X=eye(12);
for i=1:(t/12)+1
    X=[X;eye(12)];
end;
X=X(1:t,:);

% Regression coefficients and residuals
bhat=(X'*X)\(X'*y);
yhat=X*bhat;
ehat=y-yhat;           

