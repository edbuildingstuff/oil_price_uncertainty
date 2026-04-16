% STAGE2IRF.M

function [irf1hat,cumirf1hat]=stage2irf_hr(y,q)

t=length(q); 
pp=12; 

% Z is set of original regressors: eight lags of q and four lags of y
Z=[ones(1,t-pp)]; 
for i=0:pp
    Z=[Z; q(pp+1-i:t-i,1)'];
end;
Z=Z';

% y is dependent variable
y=y(pp+1:t,1);

% OLS
bhat=inv(Z'*Z)*Z'*y;
ehat=y-Z*bhat;

% Impulse response point estimate
irf1hat=bhat(2:end);
cumirf1hat=cumsum(irf1hat);
