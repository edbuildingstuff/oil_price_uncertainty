clear; clc;

%Similar to other paper - include data as it is: exchange rate,
%global eocn, quantity, US CPI, US M1 and a common factor for the
%fuel group 


% Load data
%Insert Oil Data
[mdata,mtxt] = xlsread('OilMaster(RACPrice).xlsx','Y');
%Insert Exchange Rate Data
[mdata2,mtxt2] = xlsread('OilMaster(RACPrice).xlsx','ExchangeRate');
%Insert Other data (oil production, US JLN uncertainty, Kilian econ
%activity
[mdata3,mtxt3] = xlsread('OilMaster(RACPrice).xlsx','Other');
%Insert fuel data (coal, oil, natural gas) to extract group comovement
[mdata4,mtxt4] = xlsread('OilMaster(RACPrice).xlsx','FuelGroup');

names = mtxt(1,2:end)';
tcode = mdata(1,:);
data = mdata(2:end,:);

%Extract exchange rate data 
tcodex = mdata2(1,:);
datax = mdata2(2:end,:);

%Extract data for global economic activity
tcodey = mdata3(1,1);
datay = mdata3(2:end,1);
%extract oil production data
dataq = mdata3(2:end,2); 
%extract data for US uncertainty
dataunc = mdata3(2:end,3);
%extract OECD IP production 
dataip = mdata3(2:end,4); 
%extract data for inventory
datainvt = mdata3(2:end,5);
%extract data for US M1 money stock
datam1 = mdata3(2:end,6);
%extract data for US CPI
datacpi = mdata3(2:end,7);

%Extracf fuel group data
tcodefuel = mdata4(1,:);
datafuel = mdata4(2:end,:);

%%% There are two ways I could construct forecast error
%1/ Create factors for specific groups of data
%2/ Similar to JLN i.e, create factors for all available data
    
%transform oil price data
for i=1:size(mdata,2);
transseries(:,i) = prepare_missing(data(:,i),tcode(1,i));
isstation(i,1) = RA_test(transseries(2:end,i));
end

%transform exchange rate data 
for i=1:size(mdata2,2);
transseries2(:,i)=prepare_missing(datax(:,i),tcodex(1,i));
isstation2(i,1) = adftest(transseries2(2:end,i));
end

%transform fuel data
for i=1:size(mdata4,2);
transseries3(:,i)=prepare_missing(datafuel(:,i),tcodefuel(1,i));
isstation3(i,1) = adftest(transseries3(2:end,i));
end

yt     = transseries(2:end,:); %remove NaNs
yt     = zscore(yt(:,1:end)); %only oil price data

fx = transseries2(2:end,:); %exchange rate 
fx = zscore(fx(:,1:end));

fx2 = transseries3(2:end,:); %fuel data  
fx2 = zscore(fx2(:,1:end)); 

%select commodity exchange rate: AUD, CAD, CLP, NZD, ZAR
ind=[1,2,3,4,6];
for i=1:length(ind)
xt(:,i)=fx(:,ind(i));
end

%prepare other important data identified in the literature
%
y = prepare_missing(datay,1);
y=zscore(y(2:end,1));
% unc = prepare_missing(dataunc,5);
% unc=zscore(unc(2:end,1));
quant = prepare_missing(dataq,5);
quant = zscore(quant(2:end,1));
% ip = prepare_missing(dataip,5);
% ip = zscore(ip(2:end,1));
inventory=prepare_missing(datainvt,5);
inventory = zscore(inventory(2:end,1));
m1=prepare_missing(datam1,5);
m1 = zscore(m1(2:end,1));
cpi=prepare_missing(datacpi,5);
cpi = zscore(cpi(2:end,1));

data    = [yt];
names = names;
vartype = [tcode];

[T,N]  = size(yt);
startperiod = 1973+1/12*1;
dates = startperiod+1/12*1:1/12:startperiod+1/12*(T);

% Estimate factors for fuel group 
[ef5,fhat5,lf5,vf5] = factors_em(fx2,1,2,2);
[eg5,ghat5,lg5,vg5] = factors_em(fx2.^2,1,2,2);

xt=[xt, y, quant, inventory, m1, cpi, fhat5, fhat5.^2, ghat5];
    
[T,N]  = size(yt);

%Generate forecast error 
py     = 24;
pz     = 24;

p      = max(py,pz);
q=fix(T^(1/4)); %Greene (Econometric Analysis, 7th edition, section 20.5.2, p. 960).
%q      = fix(4*(T/100)^(2/9));
ybetas = zeros(1+py,N);

for i = 1:N
    X    = [ones(T,1),mlags(yt(:,i),py)];
    reg  = nwest(yt(py+1:end,i),X(py+1:end,:),q);
    vyt(:,i)       = reg.resid; % forecast errors
    ybetas(:,i) = reg.beta;
end
% Save data
[T,N]  = size(vyt);
ybetas = ybetas';
dates_ar  = dates(end-T+1:end)';
save aroilferrors dates_ar vyt names vartype ybetas py

% Also write to .txt file for R code
dlmwrite('aroilvyt.txt',vyt,'delimiter','\t','precision',17);