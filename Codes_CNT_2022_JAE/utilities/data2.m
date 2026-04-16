% DATA.M
% clear;clc;
% sample = 2;
% index = 2;
% opu_ind = 0;

% Sample length
if sample == 0
    time=(1973+2/12:1/12:2018+6/12)'; % 1973M2-2018M6 - full sample
elseif sample == 1
    time=(1973+2/12:1/12:2008+8/12)'; % 1973M2-2008M8 - KM2014
elseif sample == 2
    time=(1981+1/12:1/12:2017+1/12)'; % 1981M1-2017M1 - commodity factor    
else
    disp('Sample = 0 uses sample 1973M1-2018M6, sample = 1 uses sample 1973M1-2009M8 and sample = 2 uses sample 1981M1-2017M1')    
end

%% Uncertainty index
if opu_ind == 0
    load opu_baseline
    opu = opu_baseline;
    
% Sample
    opudates = (1973+2/12+24/12:1/12:2018+6/12)'; % 1973M2-2018M6 - full sample + 24 lags
    opumat = [opudates,opu];    
elseif opu_ind == 1
    load opu_np
    opu = opu_np;
    
% Sample    
    opudates = (1973+2/12:1/12:2018+6/12)'; % 1973M2-2018M6 - full sample
    opumat = [opudates,opu];
end

% Cut sample    
if sample == 0
    ind1      = find(opudates == 1973+2/12+24/12); 
    ind2      = find(opudates == 2018)+6;      
elseif sample == 1
    ind1      = find(opudates == 1973+2/12+24/12); 
    ind2      = find(opudates == 2008)+8;      
elseif sample == 2
    ind1      = find(opudates == 1981)+1; 
    ind2      = find(opudates == 2017)+1;     
end
    opu = opumat(ind1:ind2,2);
    
%% EIA MER world oil production
    load worldoilproduction.txt;
    Q=worldoilproduction/10000;
    save Q.txt Q -ascii;

%% REA
if index == 0
% Kilian global real economic activity index (corrected)
    load rea_new.txt; % Jan 1973 to June 2018
    rea = rea_new(2:end,1); % Feb 1973 to June 2018
elseif index == 1
% OECD+6 Index    
    IP = xlsread('OECD_plus6_industrial_production','world_IP','B2:B727');  % Jan 1958 to June 2018
    rea      = lagn(100*log(IP),24);
    time_rea = (1958+1/12+24/12:1/12:2018+6/12)';
    reamat   = [time_rea,rea];
    ind      = find(time_rea == 1973)+1; 
    rea_new  = reamat(ind:end,2); % Jan 1973 to June 2018
    rea = rea_new(2:end,1); % Feb 1973 to June 2018
elseif index == 2
% Global commodity factor
    rea = xlsread('Global_factor_DFG','data','B3:B435');  % Jan 1981 to Jan 2017
else
    disp('Index = 0 uses kilian index, index = 1 uses OECD+6 growth rate & index = 2 uses global commodity factor')
end
%% Zhou data
load usoilstocks.txt; 
load uspetrostocks.txt; usa=uspetrostocks;
load oecdpetrostocks.txt; oecd=oecdpetrostocks;
oecd=[zeros(179,1); oecd(180:end,1)];
for i=179:-1:1
    oecd(i,1)=oecd(i+1)*(usa(i,1)/usa(i+1,1));
end;    
scale=oecd./usa; 
stocks=usoilstocks.*scale; 

% Nov 2008: $61.65
load usracimports.txt;
load cpiaucsa.txt;

rpoil=usracimports./cpiaucsa(:,3);
rpoil=49.1*rpoil/rpoil(431,1);  

% Transform data
oq = dif(log(worldoilproduction));
op = log(rpoil(2:end));
inv = dif(stocks);

% Cut sample   
time_full=(1973+2/12:1/12:2018+6/12)'; % 1973M2-2018M6 - full sample
if sample == 0
    y = [time_full,oq,rea,op,inv]; 
    ind1      = find(time_full == 1973+2/12); 
    ind2      = find(time_full == 2018)+6;      
    y = y(ind1:ind2,:);
elseif sample == 1
    y = [time_full,oq,rea,op,inv];     
    ind1      = find(time_full == 1973+2/12); 
    ind2      = find(time_full == 2008)+8;      
    y = y(ind1:ind2,:);
elseif sample == 2
    y = [time_full,oq,op,inv]; 
    ind1      = find(time_full == 1981)+1; 
    ind2      = find(time_full == 2017)+1;     
    y = [y(ind1:ind2,1:2) rea y(ind1:ind2,3:4)];
end    

%% Merge Data
if index == 0 
    dates = y(24+1:end,1); % lose first 24 obs from uncertainty index
    y = [y(24+1:end,2:end),opu];
elseif index == 1 
    dates = y(24+1:end,1); % lose first 24 obs from uncertainty index
    y = [y(24+1:end,2:end),opu];
else
    dates = y(:,1);
    y = [y(:,2:end),opu];
end
%% Plot data
% 
% for i = 1:5
%     subplot(2,3,i); plot(dates,y(:,i)); 
% end
% 
