% TABLE_HD.M
clear; clc;
% Main results
load IRFS_main_REA;

%%
% Modal model
yhat31=zeros(T,1); yhat32=zeros(T,1); yhat33=zeros(T,1); yhat34=zeros(T,1); yhat35=zeros(T,1); 
yhat41=zeros(T,1); yhat42=zeros(T,1); yhat43=zeros(T,1); yhat44=zeros(T,1); yhat45=zeros(T,1); 

% Construct RF posterior parameters for jth admissible model   
B      = (reshape(z(end,1:(1+n*p)*n),1+n*p,n))'; 
B0inv  = reshape(z(end,(1+n*p)*n+n*n+1:(1+n*p)*n+2*n*n),n,n);

% Compute structural multipliers for modal admissible model 
IRF =irfvar([B(:,1+1:end);eye(n*(p-1)) zeros(n*(p-1),n)],B0inv,p,n,T-1);

% Compute reduced-form residuals for modal model, given X and Ydep
Uhat=Ydep-X*B';

% Compute structural shocks Ehat from reduced form shocks Uhat
% Ehat=inv(B0inv)*Uhat';
Ehat=(B0inv)\(Uhat');

% Cross-multiply the weights for the effect of a given shock on the real
% oil price (given by the relevant row of IRF) with the structural shock
% in question
for i=1:T
%      yhat31(i,1)=dot(IRF(3,1:i),Ehat(1,i:-1:1));
%      yhat32(i,1)=dot(IRF(7,1:i),Ehat(2,i:-1:1));
%      yhat33(i,1)=dot(IRF(11,1:i),Ehat(3,i:-1:1));  
% HD for price    
        yhat31(i,1)=dot(IRF(3,1:i),Ehat(1,i:-1:1));
        yhat32(i,1)=dot(IRF(8,1:i),Ehat(2,i:-1:1));
        yhat33(i,1)=dot(IRF(13,1:i),Ehat(3,i:-1:1));  
        yhat34(i,1)=dot(IRF(18,1:i),Ehat(4,i:-1:1));  
        yhat35(i,1)=dot(IRF(23,1:i),Ehat(5,i:-1:1));  

% HD for inventories        
        yhat41(i,1)=dot(IRF(4,1:i),Ehat(1,i:-1:1));
        yhat42(i,1)=dot(IRF(9,1:i),Ehat(2,i:-1:1));
        yhat43(i,1)=dot(IRF(14,1:i),Ehat(3,i:-1:1));  
        yhat44(i,1)=dot(IRF(19,1:i),Ehat(4,i:-1:1));  
        yhat45(i,1)=dot(IRF(24,1:i),Ehat(5,i:-1:1));  
end

% Time line for plots (adjust time axis later)
% time=1973+2/12+p/12:1/12:2018+6/12;
sam_dat=(1973+2/12+24/12+p/12:1/12:2018+6/12)'; % new sample 

% Select dates based on Zhou paper
id1a = find(sam_dat == 1980)+1;  % 1980:M1
id1b = find(sam_dat == 1979)+1;  % 1979:M1
id2a = find(sam_dat == 1981);    % 1980:M12
id2b = find(sam_dat == 1980)+9;  % 1980:M9
id3a = find(sam_dat == 1987);    % 1986:M12 
id3b = find(sam_dat == 1986);    % 1985:M12
id4a = find(sam_dat == 1990)+10;  % 1990:M10
id4b = find(sam_dat == 1990)+5;  % 1990:M5
id5a = find(sam_dat == 2008)+6;  % 2008:M6
id5b = find(sam_dat == 2002)+7;  % 2002:M7
id6a = find(sam_dat == 2009);    % 2008:M12 
id6b = find(sam_dat == 2008)+6;  % 2008:M6
id7a = find(sam_dat == 2016);    % 2015:M12 
id7b = find(sam_dat == 2014)+6;  % 2014:M6

% Table
table_hd_price = round([yhat31(id1a)*100-yhat31(id1b)*100 yhat31(id2a)*100-yhat31(id2b)*100 yhat31(id3a)*100-yhat31(id3b)*100 yhat31(id4a)*100-yhat31(id4b)*100 yhat31(id5a)*100-yhat31(id5b)*100 yhat31(id6a)*100-yhat31(id6b)*100 yhat31(id7a)*100-yhat31(id7b)*100;
                  yhat32(id1a)*100-yhat32(id1b)*100 yhat32(id2a)*100-yhat32(id2b)*100 yhat32(id3a)*100-yhat32(id3b)*100 yhat32(id4a)*100-yhat32(id4b)*100 yhat32(id5a)*100-yhat32(id5b)*100 yhat32(id6a)*100-yhat32(id6b)*100 yhat32(id7a)*100-yhat32(id7b)*100;
                  yhat33(id1a)*100-yhat33(id1b)*100 yhat33(id2a)*100-yhat33(id2b)*100 yhat33(id3a)*100-yhat33(id3b)*100 yhat33(id4a)*100-yhat33(id4b)*100 yhat33(id5a)*100-yhat33(id5b)*100 yhat33(id6a)*100-yhat33(id6b)*100 yhat33(id7a)*100-yhat33(id7b)*100;
%                   yhat34(id1a)*100-yhat34(id1b)*100 yhat34(id2a)*100-yhat34(id2b)*100 yhat34(id3a)*100-yhat34(id3b)*100 yhat34(id4a)*100-yhat34(id4b)*100  yhat34(id5a)*100-yhat34(id5b)*100 yhat34(id6a)*100-yhat34(id6b)*100 yhat34(id7a)*100-yhat34(id7b)*100;
                  yhat35(id1a)*100-yhat35(id1b)*100 yhat35(id2a)*100-yhat35(id2b)*100 yhat35(id3a)*100-yhat35(id3b)*100 yhat35(id4a)*100-yhat35(id4b)*100 yhat35(id5a)*100-yhat35(id5b)*100 yhat35(id6a)*100-yhat35(id6b)*100 yhat35(id7a)*100-yhat35(id7b)*100])

% table_hd_inv = round([yhat41(id1a)*100-yhat41(id1b)*100 yhat41(id2a)*100-yhat41(id2b)*100 yhat41(id3a)*100-yhat41(id3b)*100 yhat41(id4a)*100-yhat41(id4b)*100 yhat41(id5a)*100-yhat41(id5b)*100 yhat41(id6a)*100-yhat41(id6b)*100 yhat41(id7a)*100-yhat41(id7b)*100;
%                 yhat42(id1a)*100-yhat42(id1b)*100 yhat42(id2a)*100-yhat42(id2b)*100 yhat42(id3a)*100-yhat42(id3b)*100 yhat42(id4a)*100-yhat42(id4b)*100 yhat42(id5a)*100-yhat42(id5b)*100 yhat42(id6a)*100-yhat42(id6b)*100 yhat42(id7a)*100-yhat42(id7b)*100;
%                 yhat43(id1a)*100-yhat43(id1b)*100 yhat43(id2a)*100-yhat43(id2b)*100 yhat43(id3a)*100-yhat43(id3b)*100 yhat43(id4a)*100-yhat43(id4b)*100 yhat43(id5a)*100-yhat43(id5b)*100 yhat43(id6a)*100-yhat43(id6b)*100 yhat43(id7a)*100-yhat43(id7b)*100;
% %                 yhat44(id1a)*100-yhat44(id1b)*100 yhat44(id2a)*100-yhat44(id2b)*100 yhat44(id3a)*100-yhat44(id3b)*100 yhat44(id4a)*100-yhat44(id4b)*100 yhat44(id5a)*100-yhat44(id5b)*100 yhat44(id6a)*100-yhat44(id6b)*100 yhat44(id7a)*100-yhat44(id7b)*100;
%                 yhat45(id1a)*100-yhat45(id1b)*100 yhat45(id2a)*100-yhat45(id2b)*100 yhat45(id3a)*100-yhat45(id3b)*100 yhat45(id4a)*100-yhat45(id4b)*100 yhat45(id5a)*100-yhat45(id5b)*100 yhat45(id6a)*100-yhat45(id6b)*100 yhat45(id7a)*100-yhat45(id7b)*100]./100)

% table_price = [Ydep(id1a,3)-Ydep(id1b,3) Ydep(id2a,3)-Ydep(id2b,3) Ydep(id3a,3)-Ydep(id3b,3) Ydep(id4a,3)-Ydep(id4b,3) Ydep(id5a,3)-Ydep(id5b,3) Ydep(id6a,3)-Ydep(id6b,3) Ydep(id7a,3)-Ydep(id7b,3)]
% table_inv = [Ydep(id1a,4)-Ydep(id1b,4) Ydep(id2a,4)-Ydep(id2b,4) Ydep(id3a,4)-Ydep(id3b,4) Ydep(id4a,4)-Ydep(id4b,4) Ydep(id5a,4)-Ydep(id5b,4) Ydep(id6a,4)-Ydep(id6b,4) Ydep(id7a,4)-Ydep(id7b,4)]

%% Check results satify narrative restrictions - debug
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

% restrictions should all equal 1
    nr1 = ((yhat31(id_90M10)-yhat31(id_90M07))>0.1); 
    nr2 = ((yhat32(id_90M10)-yhat32(id_90M06))<0.1); 
    nr3 = ((yhat33(id_79M12)-yhat33(id_79M05))+(yhat35(id_79M12)-yhat35(id_79M05))>0.20); 
    nr4 = ((yhat33(id_90M10)-yhat33(id_90M06))+(yhat35(id_90M10)-yhat35(id_90M06))>0.1);
    nr5 = ((yhat33(id_86M12)-yhat33(id_85M12))+(yhat35(id_86M12)-yhat35(id_85M12))<-0.15); 
