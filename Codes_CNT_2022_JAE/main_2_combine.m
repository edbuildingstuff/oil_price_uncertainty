% COMBINE.M
clear; clc;
addpath('utilities')
addpath('draws')

% Load admissible posterior model estimates
disp('Loading results')
ztotal=[];
for ii = 1:51
load(sprintf('oilmodelposterior_narratives_REA_1_%d',ii));
ztotal=[ztotal; z];
end

for ii = 1:4
load(sprintf('oilmodelposterior_narratives_REA_5_%d',ii));
ztotal=[ztotal; z];
end

% for ii = 1:1
% load(sprintf('oilmodelposterior_narratives_REA_2_%d',ii));
% ztotal=[ztotal; z];
% end

for ii = 1:23
load(sprintf('oilmodelposterior_narratives_REA_1_%d_extra',ii));
ztotal=[ztotal; z];
end

for ii = 1:3
load(sprintf('oilmodelposterior_narratives_REA_2_%d_extra',ii));
ztotal=[ztotal; z];
end

z = ztotal; 
M = size(z,1);

%%
z = ztotal; 
[a,I]  = sort(z(:,end),1);
z      = z(I,:);

% 100(1-alpha)% HPD credible sets
IRF11=[]; IRF12=[]; IRF13=[]; IRF14=[]; IRF15=[]; 
IRF21=[]; IRF22=[]; IRF23=[]; IRF24=[]; IRF25=[];
IRF31=[]; IRF32=[]; IRF33=[]; IRF34=[]; IRF35=[];
IRF41=[]; IRF42=[]; IRF43=[]; IRF44=[]; IRF45=[];
IRF51=[]; IRF52=[]; IRF53=[]; IRF54=[]; IRF55=[];

disp('Making IRFs')
% z=z(end-ceil((1-alpha)*size(z,1))+1:end,:);
for i=1:size(z,1)
    B      = (reshape(z(end-i+1,1:(1+n*p)*n),1+n*p,n))'; %
    %Sigma  = reshape(z(end-i+1,(1+n*p)*n+1:(1+n*p)*n+n*n),n,n);
    B0inv  = reshape(z(end-i+1,(1+n*p)*n+n*n+1:(1+n*p)*n+2*n*n),n,n);
    irf=irfvar([B(:,1+1:end);eye(n*(p-1)) zeros(n*(p-1),n)],B0inv,p,n,H);    
    IRF11 = [IRF11; irf(1,:)];
    IRF21 = [IRF21; irf(2,:)];
    IRF31 = [IRF31; irf(3,:)];
    IRF41 = [IRF41; irf(4,:)];  
    IRF51 = [IRF51; irf(5,:)];  
    
    IRF12 = [IRF12; irf(6,:)];
    IRF22 = [IRF22; irf(7,:)];
    IRF32 = [IRF32; irf(8,:)];
    IRF42 = [IRF42; irf(9,:)];
    IRF52 = [IRF52; irf(10,:)];

    IRF13 = [IRF13; irf(11,:)];
    IRF23 = [IRF23; irf(12,:)];
    IRF33 = [IRF33; irf(13,:)];
    IRF43 = [IRF43; irf(14,:)];
    IRF53 = [IRF53; irf(15,:)];
    
    IRF14 = [IRF14; irf(16,:)];
    IRF24 = [IRF24; irf(17,:)];
    IRF34 = [IRF34; irf(18,:)];
    IRF44 = [IRF44; irf(19,:)];
    IRF54 = [IRF54; irf(20,:)];

    IRF15 = [IRF15; irf(21,:)];
    IRF25 = [IRF25; irf(22,:)];
    IRF35 = [IRF35; irf(23,:)];
    IRF45 = [IRF45; irf(24,:)];
    IRF55 = [IRF55; irf(25,:)];  
end
disp('Done.')

save IRFS_main_REA IRF11 IRF12 IRF13 IRF14 IRF15 IRF21 IRF22 IRF23 IRF24 IRF25 IRF31 IRF32 IRF33 IRF34 IRF35 IRF41 IRF42 IRF43 IRF44 IRF45 IRF51 IRF52 IRF53 IRF54 IRF55 alpha p n H T Ydep X z B0inv;
