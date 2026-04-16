% -------------------------------------------------------------------------
% Compute matrix of uncertainty estimates (ar conditional mean only)
% -------------------------------------------------------------------------

% Load data
clear; clc;
load npoilferrors;
svy = load('npoilsvmeans.txt');

% Compute uncertainty
h     = 12;
[T,N] = size(vyt);
thy   = [svy(1,:).*(1-svy(2,:));svy(2,:);svy(3,:).^2];
xy    = svy(4:end-3,:);
gy    = svy(end-3:1:end,:);
ut    = zeros(T,N,h);
for j = 1:h
    for i = 1:N
        t1 = thy(1,i)*(1-thy(2,i)^j)/(1-thy(2,i));
        t2 = thy(3,i)/2*(1-thy(2,i)^(2*j))/(1-thy(2,i)^2);
        ut(:,i,j) = exp(t1+t2+thy(2,i).*xy(:,i));
    end
end

oilu(:,:) = sqrt(ut(:,1,:));
opu_np = oilu(:,1);
dates_np=dates;
%plot(dates,npoilu(:,1),dates,npoilu(:,3));
save opu_np dates_np opu_np

