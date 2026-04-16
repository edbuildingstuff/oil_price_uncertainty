
clear
clc

load opu_baseline
load opu_np
load opu_ar

figure('Name','Role of predictors')
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates_ar,opu_ar,'k--','linewidth',1.5);
plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
legend('OPU','AR only','No Predictors');

dim = [6,5];
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
print('-dpdf','OPU_RoleOfInformation');