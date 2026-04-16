clear
clc

load opu_baseline
[data,dataname] = xlsread('UncertaintyComparison.xlsx','Sheet1');

ovxdates_pre = data(:,2);
ovxdates = ovxdates_pre(~isnan(ovxdates_pre));
ovx_pre = data(:,3);
ovx = ovx_pre(~isnan(ovx_pre));

jlndates_pre =  data(:,4);
jlndates = jlndates_pre(~isnan(jlndates_pre));
jln_pre = data(:,5);
jln = jln_pre(~isnan(jln_pre));

epudates_pre = data(:,6);
epudates = epudates_pre(~isnan(epudates_pre));
epu_pre = data(:,7);
epu = epu_pre(~isnan(epu_pre));

vixdates_pre = data(:,8);
vixdates = vixdates_pre(~isnan(vixdates_pre));
vix_pre = data(:,9);
vix = vix_pre(~isnan(vix_pre));

gprdates_pre = data(:,10);
gprdates = gprdates_pre(~isnan(gprdates_pre));
gpr_pre = data(:,11);
gpr = gpr_pre(~isnan(gpr_pre));


%standardise
opu_baseline =  zscore(opu_baseline);
ovx =zscore(ovx);
jln = zscore(jln);
epu = zscore(epu);
vix = zscore(vix);
gpr = zscore(gpr);

fig=figure;
subplot(3,2,1)
plot(dates,opu_baseline,'linewidth',2,'color','b');
hold on
h1=plot(ovxdates,ovx,'linewidth',1.5,'color','r','linestyle',':');
xlim([dates(1),dates(end)]);
legend('OPU','OVX','location','northwest')
title(sprintf('Corr(OPU,OVX) = %.2f', corr(opu_baseline(length(opu_baseline)-length(ovx)+1:end),ovx)))

subplot(3,2,2)
plot(dates,opu_baseline,'linewidth',2,'color','b');
hold on
h2=plot(epudates,epu,'linewidth',1.5,'color','r','linestyle',':');
xlim([dates(1),dates(end)]);
legend('OPU','EPU','location','northwest')
title(sprintf('Corr(OPU,EPU) = %.2f', corr(opu_baseline(length(opu_baseline)-length(epu)+1:end),epu)))

subplot(3,2,3)
plot(dates,opu_baseline,'linewidth',2,'color','b');
hold on
h3=plot(vixdates,vix,'linewidth',1.5,'color','r','linestyle',':');
xlim([dates(1),dates(end)]);
legend('OPU','VIX','location','northwest')
title(sprintf('Corr(OPU,VIX) = %.2f', corr(opu_baseline(length(opu_baseline)-length(vix)+1:end),vix)))

subplot(3,2,4)
plot(dates,opu_baseline,'linewidth',2,'color','b');
hold on
h4=plot(jlndates,jln,'linewidth',1.5,'color','r','linestyle',':');
legend('OPU','JLN','location','northwest')
xlim([dates(1),dates(end)]);
title(sprintf('Corr(OPU,JLN) = %.2f', corr(opu_baseline(length(opu_baseline)-length(jln)+1:end),jln)))

subplot(3,2,5)
plot(dates,opu_baseline,'linewidth',2,'color','b');
hold on
h5=plot(gprdates,gpr,'linewidth',1.5,'color','r','linestyle',':');
legend('OPU','GPR','location','northwest')
xlim([dates(1),dates(end)]);
title(sprintf('Corr(OPU,GPR) = %.2f', corr(opu_baseline(length(opu_baseline)-length(gpr)+1:end),gpr)))



dim = [6,5];
set(gcf,'paperpositionmode','manual','paperunits','inches');
set(gcf,'papersize',dim,'paperposition',[0,0,dim]);
print('-dpdf','OPU_comparison');
