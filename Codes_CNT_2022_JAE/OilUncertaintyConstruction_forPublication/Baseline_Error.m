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
ybetas = zeros(1+py+pz*size(xt,2),N);


%Generate different predictor set for different group of commodity
for i = 1:N
    X    = [ones(T,1),mlags(yt(:,i),py),mlags(xt,pz)];
    reg  = nwest(yt(p+1:end,i),X(p+1:end,:),q);
    pass = abs(reg.tstat(py+2:end)) > 2.575; 
    keep = [ones(1,py+1)==1,pass'];       
    Xnew = X(:,keep);
    reg  = nwest(yt(p+1:end,i),Xnew(p+1:end,:),q);
    vyt(:,i)       = reg.resid; % forecast errors
    ybetas(keep,i) = reg.beta;   
    
    ybetas_noar  = [1; zeros(py,1); ybetas(py+2:end)];
    vyt_noar =  yt(p+1:end,i) - X(p+1:end,:)*ybetas_noar; 
    
    ybetas_noer = ybetas;
    ybetas_noer(26:13:end)=0;
    ybetas_noer(27:13:end)=0;
    ybetas_noer(28:13:end)=0;
    ybetas_noer(29:13:end)=0;
    ybetas_noer(30:13:end)=0;
    vyt_noer = yt(p+1:end,i) - X(p+1:end,:)*ybetas_noer; 
    
    ybetas_noy = ybetas;
    ybetas_noy(31:13:end)=0;
    vyt_noy = yt(p+1:end,i) - X(p+1:end,:)*ybetas_noy; 
    
    ybetas_noq = ybetas;
    ybetas_noq(32:13:end)=0;
    vyt_noq = yt(p+1:end,i) - X(p+1:end,:)*ybetas_noq; 
    
    ybetas_noinventory = ybetas;
    ybetas_noinventory(33:13:end)=0;
    vyt_noinventory = yt(p+1:end,i) - X(p+1:end,:)*ybetas_noinventory;
    
    ybetas_nom1 = ybetas;
    ybetas_nom1(34:13:end)=0;
    vyt_nom1 = yt(p+1:end,i) - X(p+1:end,:)*ybetas_nom1;
    
    ybetas_nocpi = ybetas;
    ybetas_nocpi(35:13:end)=0;
    vyt_nocpi = yt(p+1:end,i) - X(p+1:end,:)*ybetas_nocpi;
    
    ybetas_nocom = ybetas;
    ybetas_nocom(36:13:end)=0;
    ybetas_nocom(37:13:end)=0;
    ybetas_nocom(38:13:end)=0;
    vyt_nocom =  yt(p+1:end,i) - X(p+1:end,:)*ybetas_nocom;
    
    fmodels(:,i)   = pass; %chosen predictors
end

modeler1=sum(fmodels(1:13:end));
modeler2=sum(fmodels(2:13:end));
modeler3=sum(fmodels(3:13:end));
modeler4=sum(fmodels(4:13:end));
modeler5=sum(fmodels(5:13:end));

modely = sum(fmodels(6:13:end)); 
modelq = sum(fmodels(7:13:end)); 
modelinventory = sum(fmodels(8:13:end)); 
modelm1 = sum(fmodels(9:13:end)); 
modelcpi = sum(fmodels(10:13:end)); 

modelcom1 = sum(fmodels(11:13:end)); 
modelcom2 = sum(fmodels(12:13:end)); 
modelcom3 = sum(fmodels(13:13:end)); 



% Generate AR() errors for ft
[T,R]  = size(xt);
pf     = py;
%q      = fix(4*(T/100)^(2/9));
fbetas = zeros(R,pf+1);
for i = 1:R
   X   = [ones(T,1),mlags(xt(:,i),pf)];
   reg = nwest(xt(pf+1:end,i),X(pf+1:end,:),q);
   vft(:,i)    = reg.resid;
   fbetas(i,:) = reg.beta';
end

[T,N]  = size(vyt);
ybetas = ybetas';
ybetas_noar = ybetas_noar';
ybetas_noer = ybetas_noer';
ybetas_noy = ybetas_noy';
ybetas_noq = ybetas_noq';
ybetas_noinventory = ybetas_noinventory';
ybetas_nom1 = ybetas_nom1';
ybetas_nocpi = ybetas_nocpi';
ybetas_nocom = ybetas_nocom';


dates  = dates(end-T+1:end)';
save ferrorsoil_baseline dates vyt vft names vartype ybetas fbetas py pz pf yt xt fmodels     ...
     vyt_noar vyt_noer vyt_noy vyt_noq vyt_noinventory  vyt_nom1 vyt_nocpi  vyt_nocom ...
     ybetas_noar ybetas_noer ybetas_noy ybetas_noq ybetas_noinventory ybetas_nom1 ybetas_nocpi ybetas_nocom 
      
% Also write to .txt file for R code
dlmwrite('Oilyt_baseline.txt',vyt,'delimiter','\t','precision',17);
dlmwrite('Oilft_baseline.txt',vft,'delimiter','\t','precision',17);

dlmwrite('Oilyt_baseline_noar.txt',vyt_noar,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_noer.txt',vyt_noer,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_noy.txt',vyt_noy,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_noq.txt',vyt_noq,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_noinventory.txt',vyt_noinventory,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_nom1.txt',vyt_nom1,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_nocpi.txt',vyt_nocpi,'delimiter','\t','precision',17);
dlmwrite('Oilyt_baseline_nocom.txt',vyt_nocom,'delimiter','\t','precision',17);
