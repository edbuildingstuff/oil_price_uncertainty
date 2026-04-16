% TABLE_IRFs.M
clear; clc;

% Main results
load IRFS_main_REA;

%%
% Modal model
irf11m=IRF11(1,:); irf12m=IRF12(1,:); irf13m=IRF13(1,:); irf14m=IRF14(1,:); irf15m=IRF15(1,:);
irf21m=IRF21(1,:); irf22m=IRF22(1,:); irf23m=IRF23(1,:); irf24m=IRF24(1,:); irf25m=IRF25(1,:);
irf31m=IRF31(1,:); irf32m=IRF32(1,:); irf33m=IRF33(1,:); irf34m=IRF34(1,:); irf35m=IRF35(1,:);
irf41m=IRF41(1,:); irf42m=IRF42(1,:); irf43m=IRF43(1,:); irf44m=IRF44(1,:); irf45m=IRF45(1,:);
irf51m=IRF51(1,:); irf52m=IRF52(1,:); irf53m=IRF53(1,:); irf54m=IRF54(1,:); irf55m=IRF55(1,:);


% Transformations
irf11mt=cumsum(irf11m)*100; irf21mt=irf21m; irf31mt=irf31m*100; irf41mt=cumsum(irf41m);
irf12mt=cumsum(irf12m)*100; irf22mt=irf22m; irf32mt=irf32m*100; irf42mt=cumsum(irf42m);
irf13mt=cumsum(irf13m)*100; irf23mt=irf23m; irf33mt=irf33m*100; irf43mt=cumsum(irf43m);
irf15mt=cumsum(irf15m)*100; irf25mt=irf25m; irf35mt=irf35m*100; irf45mt=cumsum(irf45m);

% Table of results
ii = [1, 6, 12];
tab_IRFs_flow_supply = round([irf11mt(:,ii);irf21mt(:,ii);irf31mt(:,ii);irf41mt(:,ii)],2); % Flow Supply Shocks
tab_IRFs_flow_demand = round([irf12mt(:,ii);irf22mt(:,ii);irf32mt(:,ii);irf42mt(:,ii)],2); % Flow Demand Shocks
tab_IRFs_spec_demand = round([irf13mt(:,ii);irf23mt(:,ii);irf33mt(:,ii);irf43mt(:,ii)],2); % Speculative Demand Shocks
tab_IRFs_prec_demand = round([irf15mt(:,ii);irf25mt(:,ii);irf35mt(:,ii);irf45mt(:,ii)],2); % Precautionary Demand Shocks

