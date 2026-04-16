
%%
% Load data
clear; clc;
load ferrorsoil_baseline;
svf = load('Oilsvfmeans_baseline.txt');
svy = load('Oilsvymeans_baseline.txt');

% Compute objects from predictors
h   = 12;
fb  = sparse(fbetas);
thf = [svf(1,:).*(1-svf(2,:));svf(2,:);svf(3,:).^2];
xf  = svf(4:end-3,:);
gf  = svf(end-3+1:end,:);
[evf,phif] = compute_uf(xf,thf,fb,h);

% Compute uncertainty
[T,N] = size(vyt);
ut    = zeros(T,N,h);
for i = 1:N
    tic;
    yb  = sparse(ybetas(i,:));
    thy = [svy(1,i).*(1-svy(2,i));svy(2,i);svy(3,i).^2];
    xy  = svy(4:end-3,i);
    ut(:,i,:) = compute_uy(xy,thy,yb,py,evf,phif);
    fprintf('Series %d, Elapsed Time = %0.4f \n',i,toc);
end

oilu(:,:) = sqrt(ut(:,1,:));
%plot(dates,oilu_from1973_method3(:,1),dates,oilu_from1973_method3(:,3));
%save oiluncertainty_from1973_method3 dates oilu_from1973_method3

% Extract  data
opu_baseline = oilu(:,1);
save opu_baseline opu_baseline dates 


%% 
Baseline_Uncertainty_noer
Baseline_Uncertainty_noy
Baseline_Uncertainty_noq
Baseline_Uncertainty_noinventory
Baseline_Uncertainty_nom1
Baseline_Uncertainty_nocpi
Baseline_Uncertainty_nocom




