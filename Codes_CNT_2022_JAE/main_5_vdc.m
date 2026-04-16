% TABLE_VDC.M
clear; clc;

% Main results
load IRFS_main_REA;

%%
% Modal model (last model after sorting in combine.m)
A      = (reshape(z(end,1:(1+n*p)*n),1+n*p,n))'; A=[A(:,2:end);eye(n*(p-1)) zeros(n*(p-1),n)];
B0inv  = reshape(z(end,(1+n*p)*n+n*n+1:(1+n*p)*n+2*n*n),n,n);

% Structural forecast error variance Decomposition at horizon h, 
% where h=600 ~ h=infinity
h=600;   
J=[eye(n,n) zeros(n,n*(p-1))];
TH1=J*A^0*J'; TH=TH1*B0inv; TH=TH'; TH2=(TH.*TH); TH3=TH2;
for i=2:h
    TH=J*A^(i-1)*J'*B0inv; TH=TH'; TH2=(TH.*TH); TH3=TH3+TH2;
end
TH4=sum(TH3);
VC=zeros(n,n);
for j=1:n
    VC(j,:)=TH3(j,:)./TH4;
end

% VDC in percentage terms at horizon h
% Columns refer to shocks j=1,...,q that explain any given variable
% Rows refer to variables whose variation is to be explained
% Here we care about the last row (the stock return variable) and all
% columns (i.e., the contribution of each shock)
FEVD = VC'*100;
% disp(FEVD)

% Forecast error variance decomposition at horizon h for real price of oil
disp('VDC for real price of oil (%)')
disp(round(FEVD(3,:),2))

% % Forecast error variance decomposition at horizon h for inventories
% disp('VDC for inventories (%)')
% disp(round(FEVD(4,:),2))


%% 
% Structural forecast error variance Decomposition at horizon h, 
% where h=600 ~ h=infinity
h=600;   
sFEVD = NaN(h,n);
for hor = 1:h    
J=[eye(n,n) zeros(n,n*(p-1))];
TH1=J*A^0*J'; TH=TH1*B0inv; TH=TH'; TH2=(TH.*TH); TH3=TH2;
for i=2:hor
    TH=J*A^(i-1)*J'*B0inv; TH=TH'; TH2=(TH.*TH); TH3=TH3+TH2;
end
TH4=sum(TH3);
VC=zeros(n,n);
for j=1:n
    VC(j,:)=TH3(j,:)./TH4;
end

% VDC in percentage terms at horizon h
% Columns refer to shocks j=1,...,q that explain any given variable
% Rows refer to variables whose variation is to be explained
% Here we care about the last row (the stock return variable) and all
% columns (i.e., the contribution of each shock)
FEVD = VC'*100;
% disp(FEVD)

% Forecast error variance decomposition at horizon h for real price of oil
sFEVD(hor,:) = (round(FEVD(3,:),2));
end


%%
% figure
% plot(sFEVD(1:200,[1,2,3,5]),'LineWidth',1.5);
% legend('Flow Supply Shock','Flow Demand Shock','Speculative Demand Shock','Precautionary Demand shock','Location','SouthEast')
% xlabel('Forecast Horizon (months)')
% ylabel('Variance Contribution (percent)')
% ylim([0,50])
% xticks(0:25:200)
%%

figure
plot(sFEVD(1:200,1),'LineWidth',1.5,'Color','b');
hold on
plot(sFEVD(1:200,2),'LineWidth',1.5,'LineStyle',':','Color','r');
plot(sFEVD(1:200,3),'LineWidth',1.5,'LineStyle','--','Color','m');
plot(sFEVD(1:200,5),'LineWidth',0.5,'Color','k','LineStyle','--','Marker','x');
legend('Flow Supply Shock','Flow Demand Shock','Speculative Demand Shock','Precautionary Demand shock','Location','SouthEast')
xlabel('Forecast Horizon (months)')
ylabel('Variance Contribution (percent)')
ylim([0,50])
xticks(0:25:200)





