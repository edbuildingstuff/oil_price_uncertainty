
%% %% 3/ Oil 
clear; clc;

mc=1;
[mdata,mtxt] = xlsread('OilMaster(RACPrice).xlsx','Y');
names = mtxt(1,2:end)';
data = mdata(2:end,:);

tcode(1,1)=5;
tcode(1,2)=5;
tcode(1,3)=1;

for i=1:size(mdata,2);
transseries(:,i) = prepare_missing(data(:,i),tcode(1,i));
isstation(i,1) = RA_test(transseries(3:end,i));
end

yt          = transseries(2:end,1); %remove NaNs
T           = size(yt,1);

data    = [yt];
names = names;
vartype = [tcode];

%Define start period %Start period is M01 1990; thus 1990
startperiod = 1973+1/12*1;
dates = startperiod+1/12*1:1/12:startperiod+1/12*(T);
% Generate forecast errors for yt
yt     = zscore(yt(:,1:end)); % only the macro data
vyt    = yt;
% % Save data
% [T,N]  = size(vyt);
% dates  = 1900+(59:1/12:112-1/12)';
% dates  = dates(end-T+1:end);
save npoilferrors dates vyt names vartype

% Also write to .txt file for R code
dlmwrite('npoilvyt.txt',vyt,'delimiter','\t','precision',17);