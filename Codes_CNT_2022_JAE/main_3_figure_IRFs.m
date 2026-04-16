% FIGUREA1.M
clear;
% Main results
load IRFS_main_REA;

alpha = 0; % posterior of IRFs are already trimmed

%%
A=size(IRF11,1);
irf11m=IRF11(1,:); irf12m=IRF12(1,:); irf13m=IRF13(1,:); irf14m=IRF14(1,:); irf15m=IRF15(1,:);
irf21m=IRF21(1,:); irf22m=IRF22(1,:); irf23m=IRF23(1,:); irf24m=IRF24(1,:); irf25m=IRF25(1,:);
irf31m=IRF31(1,:); irf32m=IRF32(1,:); irf33m=IRF33(1,:); irf34m=IRF34(1,:); irf35m=IRF35(1,:);
irf41m=IRF41(1,:); irf42m=IRF42(1,:); irf43m=IRF43(1,:); irf44m=IRF44(1,:); irf45m=IRF45(1,:);
irf51m=IRF51(1,:); irf52m=IRF52(1,:); irf53m=IRF53(1,:); irf54m=IRF54(1,:); irf55m=IRF55(1,:);

% Plot the modal model IRF estimate along with joint HPD regions 
fs = 10; % font size
figure('Name','IRFs')
%--------------------------------------------------------------------------
% Flow supply shock
subplot(4,4,1)
% plot([0:H],cumsum(irf11m)*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF11(i,:))*100,'r-');
end
plot([0:H],cumsum(irf11m)*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Oil production','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow supply shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -2 1])
hold

subplot(4,4,2)
% plot([0:H],irf21m,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF21(i,:),'r-');
end
plot([0:H],irf21m,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real activity','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow supply shock','fontsize',fs);
% set(gca,'XTick',[0 5 10 15])
axis([0 17 -15 5])
set(gca,'XTick',[0 5 10 15])
% axis([0 17 -1 1])
hold

subplot(4,4,3)
% plot([0:H],irf31m*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF31(i,:)*100,'r-');
end
plot([0:H],irf31m*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real price of oil','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow supply shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -5 10])
hold

subplot(4,4,4)
% plot([0:H],cumsum(irf41m),'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF41(i,:)),'r-');
end
plot([0:H],cumsum(irf41m),'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Inventories','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow supply shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -40 40])
hold

%--------------------------------------------------------------------------
% Flow demand shock
subplot(4,4,5)
% plot([0:H],cumsum(irf12m)*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF12(i,:))*100,'r-');
end
plot([0:H],cumsum(irf12m)*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Oil production','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -1 2])
hold


subplot(4,4,6)
% plot([0:H],irf22m,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF22(i,:),'r-');
end
plot([0:H],irf22m,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real activity','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow demand shock','fontsize',fs);
% set(gca,'XTick',[0 5 10 15])
axis([0 17 -5 20])
set(gca,'XTick',[0 5 10 15])
% axis([0 17 -1 2])
hold

subplot(4,4,7)
% plot([0:H],irf32m*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF32(i,:)*100,'r-');
end
plot([0:H],irf32m*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real price of oil','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -5 15])
hold

subplot(4,4,8)
% plot([0:H],cumsum(irf42m),'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF42(i,:)),'r-');
end
plot([0:H],cumsum(irf42m),'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Inventories','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Flow demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -40 40])
hold

%--------------------------------------------------------------------------
% Speculative demand shock
subplot(4,4,9)
% plot([0:H],cumsum(irf13m)*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF13(i,:))*100,'r-');
end
plot([0:H],cumsum(irf13m)*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Oil production','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Speculative demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -1 2])
hold

subplot(4,4,10)
% plot([0:H],irf23m,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF23(i,:),'r-');
end
plot([0:H],irf23m,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real activity','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Speculative demand shock','fontsize',fs);
% set(gca,'XTick',[0 5 10 15])
axis([0 17 -15 15])
set(gca,'XTick',[0 5 10 15])
% axis([0 17 -1 1])
hold

subplot(4,4,11)
% plot([0:H],irf33m*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF33(i,:)*100,'r-');
end
plot([0:H],irf33m*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real price of oil','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Speculative demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -5 10])
hold

subplot(4,4,12)
% plot([0:H],cumsum(irf43m),'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF43(i,:)),'r-');
end
plot([0:H],cumsum(irf43m),'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Inventories','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Speculative demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -40 40])
hold

%--------------------------------------------------------------------------
% Precautionary demand shock
subplot(4,4,13)
% plot([0:H],cumsum(irf15m)*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF15(i,:))*100,'r-');
end
plot([0:H],cumsum(irf15m)*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Oil production','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Precautionary demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -2 2])
hold

subplot(4,4,14)
% plot([0:H],irf25m,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF25(i,:),'r-');
end
plot([0:H],irf25m,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real activity','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Precautionary demand shock','fontsize',fs);
% set(gca,'XTick',[0 5 10 15])
axis([0 17 -15 10])
set(gca,'XTick',[0 5 10 15])
% axis([0 17 -2 1])
hold

subplot(4,4,15)
% plot([0:H],irf35m*100,'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],IRF35(i,:)*100,'r-');
end
plot([0:H],irf35m*100,'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Real price of oil','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Precautionary demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -10 10])
hold

subplot(4,4,16)
% plot([0:H],cumsum(irf45m),'k-',[0:H],zeros(H+1,1),'k-','linewidth',2);
hold;
for i=1:1:ceil((1-alpha)*A)
    plot([0:H],cumsum(IRF45(i,:)),'r-');
end
plot([0:H],cumsum(irf45m),'k-','linewidth',2);
plot([0:H],zeros(H+1,1),'k:','linewidth',1);
ylabel('Inventories','fontsize',fs);
xlabel('Months','fontsize',fs);
title('Precautionary demand shock','fontsize',fs);
set(gca,'XTick',[0 5 10 15])
axis([0 17 -40 40])
hold

% save ikstyle_update_short irf11m irf12m irf13m irf14m irf21m irf22m irf23m irf24m irf31m irf32m irf33m irf34m irf41m irf42m irf43m irf44m;
