clear
clc

load opu_baseline_nocom
load opu_baseline_nocpi
load opu_baseline_noer
load opu_baseline_noinventory
load opu_baseline_nom1
load opu_baseline_noq
load opu_baseline_noy
load opu_baseline
load opu_np


subplot(3,3,1)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_nocom,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('Exchange Rates')

subplot(3,3,2)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_noy,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('Real Econ. Act.')

subplot(3,3,3)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_noq,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('Quantity')

subplot(3,3,4)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_noinventory,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('Inventory')

subplot(3,3,5)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_nom1,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('U.S. M1')

subplot(3,3,6)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_nom1,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('U.S. CPI')

subplot(3,3,7)
plot(dates,opu_baseline,'b-','linewidth',2);
hold on
plot(dates,opu_baseline_nocom,'k--','linewidth',1.5)
%plot(dates_np,opu_np,'r:','linewidth',1);
xlim([dates(1),dates(end)]);
title('Excess Comovement')

subplot(3,3,8:9)
plot(0,0,'b-','linewidth',2)
hold on
%plot(0,0,'r:','linewidth',1)
plot(0,0,'k--','linewidth',1.5)
axis off
legend('OPU','OPU without a particular predictor')
%legend('OPU','No Predictor','No particular predictor')

dim = [6,5];
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
print('-dpdf','OPU_IndividualContribution');