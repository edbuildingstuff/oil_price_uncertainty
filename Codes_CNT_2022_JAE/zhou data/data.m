% DATA.M
% sample dates
if sample == 0
    sam_dat=(1973+2/12:1/12:2018+6/12)'; % 1973M2-2018M6 
elseif sample == 1
    sam_dat=(1973+2/12:1/12:2018+6/12)'; % 1973M2-2008M8
else
    disp('Sample must be a binary value: sample = 0 uses sample 1973M1 until 2018M6 & sample = 1 uses sample 1973M1 until 2009M8')    
end

%% Uncertainty index
    load opu_baseline
    opu = opu_baseline;
%     
%     load opu_robust12lags
%     opu = opu_robust12lags;    

%% EIA MER world oil production
load worldoilproduction.txt;
Q=worldoilproduction/10000;
save Q.txt Q -ascii;

%% REA
if index == 0
% Kilian global real economic activity index (corrected)
load rea_new.txt; % Jan 1973 to June 2018
elseif index == 1
IP = xlsread('OECD_plus6_industrial_production','world_IP','B2:B727');  % Jan 1958 to June 2018
% time_IP = (1958+1/12:1/12:2018+6/12)';
rea      = lagn(100*log(IP),24);
time_rea = (1958+1/12+24/12:1/12:2018+6/12)';
reamat   = [time_rea,rea];
ind      = find(time_rea == 1973)+1; 
rea_new  = reamat(ind:end,2); % Jan 1973 to June 2018
else
    disp('Index must be a binary value: index = 0 uses kilian index and index = 1 uses OECD+6 growth rate ')
end
%% EIA MER stock data
load usoilstocks.txt; 
load uspetrostocks.txt; usa=uspetrostocks;
load oecdpetrostocks.txt; oecd=oecdpetrostocks;
oecd=[zeros(179,1); oecd(180:end,1)];
for i=179:-1:1
    oecd(i,1)=oecd(i+1)*(usa(i,1)/usa(i+1,1));
end
scale=oecd./usa; 
stocks=usoilstocks.*scale; 

% Nov 2008: $61.65
load usracimports.txt;
load cpiaucsa.txt;

rpoil=usracimports./cpiaucsa(:,3);
rpoil=49.1*rpoil/rpoil(431,1);  

% y=[dif(log(worldoilproduction)) rea_new(2:end,1) log(rpoil(2:end)) dif(stocks)];

y=[sam_dat dif(log(worldoilproduction)) rea_new(2:end,1) log(rpoil(2:end)) dif(stocks)];

%% Cut sample to match uncertainty index
y = y(24+1:end,2:end);   % lose first 24 obs in creation of opu
% y = y(12+1:end,2:end); % lose first 24 obs in creation of opu
% sam_dat=(1973+2/12+24/12:1/12:2018+6/12)'; % new sample

%% Merge Data
y = [y,opu];

%% Plot data

for i = 1:5
    subplot(2,3,i); plot(dates,y(:,i)); 
end

