% MAIN_PARALLEL.M
% Updated code for Kilian-Murphy (2014) model using Inoue and Kilian
% (2013, 2018) method of Bayesian inference. Updated sample period.
% Supply elasticity bound of 0.04.
% Narrative restrictions as in Zhou (2020), modified to take into account
% seperation of precautionary and speculative demand shocks.
% OPU index as in Cross, Nguyen and Tran (2022)
clc; clear all;
close all;
STREAM = RandStream.getGlobalStream;
% rng('shuffle')

% Set paths
addpath('OPU')
addpath('utilities')
addpath('zhou data')

%% Controls
H     = 17;       % Maximum horizon 
p     = 24;       % VAR order 
% M      = 50000; % Number of RF posterior draws
rep   = 20000;    % Number of rotations per RF posterior draw
alpha = 0.32;     % 100(1-alpha)% credible sets: 68% interval as baseline
index = 0;        % 0 uses kilian index and 1 uses OECD+6 growth rate 
sample = 0;       % 0 uses sample 1973M1 until 2018M6 as in Zhou (2019, JAE)
                  % 1 uses sample 1973M1 until 2009M8 as in Kilian and Murphey (2014, JAE)
store_draws = 100; % no. of accepted draws 

%% Set up MCMC
% Data for conditional MLE/LS estimator for VAR model with intercept
data; 
[t,n]=size(y); T=t-p;
% sam_dat=(1973+2/12+24/12:1/12:2018+6/12)'; % Sample
% sam_dat=(1973+2/12+24/12+p/12:1/12:2018+6/12)'; % Sample removing lags

% Preliminaries for computing elasticity of oil demand in use
%(outside loop because they are the same in each loop)
load Q.txt 
Q_1=(10000*Q(2:end,1))*30/1000;
DSbar=mean(y(:,4));

% Remove seasonal variation by fitting seasonal dummies equation by
% equation
y(:,1)=deseasonal(y(:,1));
y(:,2)=deseasonal(y(:,2));
y(:,3)=deseasonal(y(:,3));
y(:,4)=deseasonal(y(:,4));

Y=y(p:t,:);	
for i=1:p-1
 	Y=[Y y(p-i:t-i,:)];
end
X=[ones(T,1) Y(1:t-p,:)];
Bhat=(X'*X)\(X'*y(p+1:t,:));
Sigmahat = (y(p+1:t,:)-X*Bhat)'*(y(p+1:t,:)-X*Bhat)/T;
Ydep=y(p+1:t,:);

% Prior parameters
Bbar0=zeros(p*n+1,n);
nu0=0;
N0=zeros(n*p+1,n*p+1);
S0=zeros(n,n); 

% Posterior parameters
nuT   = T+nu0;
NT    = N0+X'*X;
BbarT = (NT)\(N0*Bbar0+X'*X*Bhat);
ST    = (nu0/nuT)*S0+(T/nuT)*Sigmahat+(1/nuT)*(Bhat-Bbar0)'*N0*(NT\X')*X*(Bhat-Bbar0);
EvecB = reshape(BbarT,n*(n*p+1),1); % Posterior mean of vec(B)

% disp('Itializing Sigma')
% Sigmamat = [];
% for i=1:M
%     RANTR=chol(inv(ST))'*randn(n,nuT)/sqrt(nuT);
%     Sigma=inv(RANTR*RANTR');
%     Sigmamat = [Sigmamat; reshape(Sigma,1,n*n)];    
% end

%% Get dates for narrative restrictions
% sample removing lags
sam_dat=(1973+2/12+24/12+p/12:1/12:2018+6/12)'; 
% flow supply shocks raise the log real price of oil by at least 0.1 between July (M7) and October (M10) of 1990
% flow demand shocks raise the log real price of oil by at most 0.1 between June (M6) and October (M10) of 1990
% storage demand shocks raised the log real price of oil by at least 0.1 between June 1990 and October 1990
    id_90M10 = find(sam_dat == 1990)+10; % October 1990:M10
    id_90M06 = find(sam_dat == 1990)+6;  % June 1990:M6
    id_90M07 = find(sam_dat == 1990)+7;  % July 1990:M7

% storage demand shocks raise the log real price of oil by at least 0.2 between May and December 1979
    id_79M05 = find(sam_dat == 1979)+5;   % May 1979:M5
    id_79M12 = find(sam_dat == 1979)+12;  % December 1979:M12
% storage demand lower the log real price of oil by at least 0.15 between December 1985 and 1986
    id_85M12 = find(sam_dat == 1985)+12;  % December 1985:M12
    id_86M12 = find(sam_dat == 1986)+12;  % December 1986:M12

%% Evaluations of the posterior pdf at posterior draws
disp('Starting MCMC')
z = [];
count_S1 = 0;
count_S2 = 0;
count_S3 = 0;
count_S4 = 0;
count_S5 = 0;
count_draws1 = 0;

i = 0;
% for i=1:M
while size(z,1)<store_draws
i = i + 1;
%     disp(i)
%     Sigma  = reshape(Sigmamat(i,1:n*n),n,n);
    RANTR  = chol(ST\speye(n))'*randn(n,nuT)/sqrt(nuT);
    Sigma  = (RANTR*RANTR')\speye(n);
    VvecB  = kron(Sigma,NT\speye(n*p+1));       
    vecB   = EvecB+chol(VvecB)'*randn(n*(n*p+1),1);
    B      = (reshape(vecB,1+n*p,n))'; 
    A      = chol(Sigma)';        
    
    % For each posterior draw simulate R rotations and keep all admissible
    % rotations
    zr=zeros(rep,p*n*n+n+n^2+n^2+1);
    root=max(abs(eig([B(:,1+1:end);eye(n*(p-1)) zeros(n*(p-1),n)])));
    if root<0.99101
%% S1: Sign restrictions       
        %parfor r=1:rep  
        for r=1:rep  

           [U,R]  = qr(randn(n,n));
           for j=1:n
               if R(j,j)<0
                   U(:,j)=-U(:,j);
               end
           end
           Atilde = A*U;
               
           % Reshuffle columns of candidate draw as needed, given Atilde
%            shuffle
           shuffle_mod_unc_nosup
           
           % Rearrange columns of B0inv as needed
%            if identot==111
           if identot==1111
% disp('S1 check')
count_S1 = count_S1 + 1;               
%                return
%% S2: Elasticity restrictions        
            % Demand and supply elasticities for candidate solution
            % after reshuffling columns
%             eta_use_t=(((Q_1*irfs1)-irfs4)./(Q_1-DSbar))/(irfs3);
%             eta_use=mean(eta_use_t);  
%             eta_s1=irsd1/irsd3;
%             eta_s2=irfd1/irfd3;
            eta_use_t=((Q_1*B0inv(1,1)-B0inv(4,1))./(Q_1-DSbar))/(B0inv(3,1));
            eta_use=mean(eta_use_t);  
            eta_s1a=B0inv(1,3)/B0inv(3,3); %supply response to speculative demand shock 
            eta_s1b=B0inv(1,5)/B0inv(3,5); %supply response to uncertainty demand shock 
            eta_s2=B0inv(1,2)/B0inv(3,2); %supply response to flow demand shock
            
%             clear irf ir Atilde identot s1is s2is s3is s4is irfs1 irfs3 irfs4 irfs3 irsd1 irsd3 irfd1 irfd3; 
 
            % Impose elasticity bound and narrative sign restrictions
%             if (eta_s1<0.04)&&(eta_s2<0.04)&&(eta_use<0)&&(eta_use>-0.8)     
            if (eta_s1a<0.04)&&(eta_s1b<0.04)&&(eta_s2<0.04)&&(eta_use<0)&&(eta_use>-0.8)  
% disp('S2 check')
count_S2 = count_S2 + 1;                
%                return           
%% S3: Dynamic restrictions
            % Candidate solution of structural model
            irf=IRvar([B(:,1+1:end);eye(n*(p-1)) zeros(n*(p-1),n)],B0inv,n,p,H);
            ir=cumsum(irf,3);   % Cumulative impulse responses (in third dimension, which is the horizon)
        
            % Normalization to take sign switch into account
            co=zeros(13,n*n);
               for sh=1:n
                   for k=1:1+H
                       co(k,1+n*(sh-1))=ir(1,sh,k)/ir(1,sh,1); 
                       co(k,2+n*(sh-1))=irf(2,sh,k)/irf(1,sh,1);
                       co(k,3+n*(sh-1))=irf(3,sh,k)/irf(1,sh,1);
                       co(k,4+n*(sh-1))=ir(4,sh,k)/ir(1,sh,1);
                       co(k,5+n*(sh-1))=ir(5,sh,k)/ir(1,sh,1);
                   end
               end
            if sum(sign(co(1:12,1))>=0)==12 && sum(sign(co(1:12,2))>=0)==12 && sum(sign(co(1:12,3))<=0)==12 ...
             && sum(sign(co(1:12,13))>=0)==12 && sum(sign(co(1:12,14))>=0)==12 ...
             && sum(sign(co(1:12,23))>=0)==12 && sum(sign(co(1:12,24))>=0)==12 ...
             && sum(sign(co(1:12,22))<=0)==12
             
% disp('S3 check')
count_S3 = count_S3 + 1;

%                return
%% S4: narrative restrictions
                % All shocks should be normalized, so they raise the real oil price.
                    B0inv(:,1)=-B0inv(:,1);       
                % Narrative sign restrictions
                % Compute reduced-form residuals for candidate model and recover
                % structural shocks
                Uhat=Ydep-X*B'; 
%                 Ehat=inv(B0inv)*Uhat';
                Ehat=(B0inv)\(Uhat'); % structural shocks
                
                % Cross-multiply the weights for the effect of a given shock on the real
                % oil price (given by the relevant row of IRF) with the structural shock
                % in question
                IRF=irfvar([B(:,1+1:end);eye(n*(p-1)) zeros(n*(p-1),n)],B0inv,p,n,t-p-1);
                yhat1=zeros(t-p,1); yhat2=zeros(t-p,1); yhat3=zeros(t-p,1); yhat5=zeros(t-p,1);
                for ii=1:t-p
%                     yhat1(ii,:)=dot(IRF(3,1:ii),Ehat(1,ii:-1:1));
%                     yhat2(ii,:)=dot(IRF(7,1:ii),Ehat(2,ii:-1:1));
%                     yhat3(ii,:)=dot(IRF(11,1:ii),Ehat(3,ii:-1:1));  
%                     yhat4(ii,:)=dot(IRF(15,1:ii),Ehat(4,ii:-1:1));  
                    yhat1(ii,:)=dot(IRF(3,1:ii),Ehat(1,ii:-1:1));  % flow supply shock effect on price of oil
                    yhat2(ii,:)=dot(IRF(8,1:ii),Ehat(2,ii:-1:1));  % flow demand shock effect on price of oil
                    yhat3(ii,:)=dot(IRF(13,1:ii),Ehat(3,ii:-1:1)); % speculative demand shock effect on price of oil
%                     yhat4(ii,:)=dot(IRF(18,1:ii),Ehat(4,ii:-1:1)); % residual
                    yhat5(ii,:)=dot(IRF(23,1:ii),Ehat(5,ii:-1:1)); % uncertainty demand shock effect on price of oil
                end
                % Check the narrative sign restrictions
%                 if (yhat3(59)-yhat3(52))>0.20 & (yhat2(189)-yhat2(185))<0.1 & (yhat3(189)-yhat3(185))>0.1  & (yhat3(143)-yhat3(131))<-0.15 & (yhat1(189)-yhat1(186))>0.1                     
                nr1 = ((yhat1(id_90M10)-yhat1(id_90M07))>0.1); 
                nr2 = ((yhat2(id_90M10)-yhat2(id_90M06))<0.1); 
                nr3 = ((yhat3(id_79M12)-yhat3(id_79M05))+(yhat5(id_79M12)-yhat5(id_79M05))>0.20); 
                nr4 = ((yhat3(id_90M10)-yhat3(id_90M06))+(yhat5(id_90M10)-yhat5(id_90M06))>0.1);
                nr5 = ((yhat3(id_86M12)-yhat3(id_85M12))+(yhat5(id_86M12)-yhat5(id_85M12))<-0.15); 
                
                % Check the narrative sign restrictions
                if nr1 && nr2 && nr3 && nr4 && nr5
%                     disp('S4 check')
                    count_S4 = count_S4 + 1;
                       post=irfpdf(B0inv,EvecB,NT,nuT,ST,vecB);    
                       zr(r,:) = [vecB' reshape(Sigma,1,n*n) reshape(B0inv,1,n*n) post];
                end  
            end
            end % end if statement for elasticity bound   
           end % end check of other restrictions
        end % end r-loop
    end % end if statement for root
  
  % Store admissibile models
  for rr=rep:-1:1
      if all(zr(rr,:))~=0
            z = [z; zr(rr,:)]; 
            disp([num2str( i ), ' draws completed']); 
            disp([num2str( size(z,1) ), ' accepted draws']); 
      end
  end
% %% counter
%     if mod(i,10000) == 0
%       disp([num2str( i ), ' loops completed']); 
%       disp([num2str( size(z,1) ), ' accepted draws']); 
%       fprintf('%d accepted static sign restrictions\n',count_S1)
%       fprintf('%d accepted magnitude restrictions\n',count_S2)
%       fprintf('%d accepted dynamic sign restrictions\n',count_S3) 
%       fprintf('%d accepted narrative sign restrictions\n',count_S4) 
%     end
end  % end i-loop
% save oilmodelposterior_a alpha M H Ydep X rep z T n p;

M = size(z,1);
rnd = rng;
save oilmodelposterior_narratives_100 alpha M H Ydep X rep z T n p rnd;
