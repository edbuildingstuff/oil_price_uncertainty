% Indicator variables for whether sign restrictions hold in
% candidate solution for impulse responses
s1is=0; s2is=0; s3is=0; s4is=0;  s5is=0;
        
% Check signs of impulse responses for each of the first four shocks

% 1. Flow supply shock: +,+,-,?,? && uncertainty impact
if Atilde(1,1)>0 && Atilde(2,1)>0 && Atilde(3,1)<0 && Atilde(5,1)<max(Atilde(5,:))
            s1is=100;
elseif Atilde(1,2)>0 && Atilde(2,2)>0 && Atilde(3,2)<0 && Atilde(5,2)<max(Atilde(5,:))
            s2is=100;
elseif Atilde(1,3)>0 && Atilde(2,3)>0 && Atilde(3,3)<0 && Atilde(5,3)<max(Atilde(5,:))
            s3is=100;
elseif Atilde(1,4)>0 && Atilde(2,4)>0 && Atilde(3,4)<0 && Atilde(5,4)<max(Atilde(5,:))
            s4is=100;
elseif Atilde(1,5)>0 && Atilde(2,5)>0 && Atilde(3,5)<0 && Atilde(5,5)<max(Atilde(5,:))
            s5is=100;
end
         
% 2. Storage (or speculative) demand shock: +,-,+,+,? && uncertainty impact
% is smaller than uncertainty case
if Atilde(1,1)>0 && Atilde(2,1)<0 && Atilde(3,1)>0 && Atilde(4,1)>0 && Atilde(5,1)<max(Atilde(5,:)) 
           s1is=10;
elseif Atilde(1,2)>0 && Atilde(2,2)<0 && Atilde(3,2)>0 && Atilde(4,2)>0 && Atilde(5,2)<max(Atilde(5,:)) 
           s2is=10;
elseif Atilde(1,3)>0 && Atilde(2,3)<0 && Atilde(3,3)>0 && Atilde(4,3)>0  && Atilde(5,3)<max(Atilde(5,:)) 
           s3is=10;
elseif Atilde(1,4)>0 && Atilde(2,4)<0 && Atilde(3,4)>0 && Atilde(4,4)>0 && Atilde(5,4)<max(Atilde(5,:)) 
           s4is=10;
elseif Atilde(1,5)>0 && Atilde(2,5)<0 && Atilde(3,5)>0 && Atilde(4,5)>0 && Atilde(5,5)<max(Atilde(5,:)) 
           s5is=10;
end
           
% 3. Flow demand shock: +,+,+,?,?
if Atilde(1,1)>0 && Atilde(2,1)>0 && Atilde(3,1)>0 
           s1is=1;
elseif Atilde(1,2)>0 && Atilde(2,2)>0 && Atilde(3,2)>0
           s2is=1;
elseif Atilde(1,3)>0 && Atilde(2,3)>0 && Atilde(3,3)>0 
           s3is=1; 
elseif Atilde(1,4)>0 && Atilde(2,4)>0 && Atilde(3,4)>0 
           s4is=1; 
elseif Atilde(1,5)>0 && Atilde(2,5)>0 && Atilde(3,5)>0 
           s5is=1; 
end  
     
% 4. Uncertainty demand shock: ?,-,+,+,+ & larger than speculative shock
if  Atilde(2,1)<0 && Atilde(3,1)>0 && Atilde(4,1)>0 && Atilde(5,1)>0 && Atilde(5,1)==max(Atilde(5,:))  
           s1is=1000;
elseif  Atilde(2,2)<0 && Atilde(3,2)>0 && Atilde(4,2)>0 && Atilde(5,2)>0 && Atilde(5,2)==max(Atilde(5,:))  
           s2is=1000;
elseif  Atilde(2,3)<0 && Atilde(3,3)>0 && Atilde(4,3)>0 && Atilde(5,3)>0 && Atilde(5,3)==max(Atilde(5,:))  
           s3is=1000;
elseif  Atilde(2,4)<0 && Atilde(3,4)>0 && Atilde(4,4)>0 && Atilde(5,4)>0 && Atilde(5,4)==max(Atilde(5,:))  
           s4is=1000;
elseif  Atilde(2,5)<0 && Atilde(3,5)>0 && Atilde(4,5)>0 && Atilde(5,5)>0 && Atilde(5,5)==max(Atilde(5,:))  
           s5is=1000;
end

% Indicator for whether sign restrictions hold
identot=s1is+s2is+s3is+s4is+s5is;

if identot==1111 % signs accepted
    % Rearrange columns of B0inv as needed
               B0inv=zeros(n,n);  
        
              % Shock 1
               if s1is==100 % supply shock
                   B0inv(:,1)=Atilde(:,1);
               elseif s1is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,1);
               elseif s1is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,1);                   
               elseif s1is==0 % residual
                   B0inv(:,4)=Atilde(:,1);
               elseif s1is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,1);
               end

               % Shock 2
               if s2is==100 % supply shock
                   B0inv(:,1)=Atilde(:,2);
               elseif s2is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,2);
               elseif s2is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,2);    
               elseif s2is==0 % residual
                   B0inv(:,4)=Atilde(:,2);
               elseif s2is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,2);
               end

               % Shock 3
               if s3is==100 % supply shock
                   B0inv(:,1)=Atilde(:,3);
               elseif s3is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,3);
               elseif s3is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,3);    
               elseif s3is==0 % residual
                   B0inv(:,4)=Atilde(:,3);
               elseif s3is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,3);
               end

               % Shock 4
               if s4is==100 % supply shock
                   B0inv(:,1)=Atilde(:,4);
               elseif s4is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,4);
               elseif s4is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,4);
               elseif s4is==0 % residual
                   B0inv(:,4)=Atilde(:,4);                   
               elseif s4is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,4);
               end

               % Shock 5
               if s5is==100 % supply shock
                   B0inv(:,1)=Atilde(:,5);
               elseif s5is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,5);
               elseif s5is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,5);
               elseif s5is==0 % residual
                   B0inv(:,4)=Atilde(:,5);                   
               elseif s5is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,5);
               end               
         
else
% take negative of impact matrix and check that
Atilde = -Atilde;
s1is=0; s2is=0; s3is=0; s4is=0; s5is=0;                    
        
% Check signs of impulse responses for each of the first three shocks

% 1. Flow supply shock: +,+,-,?,?
if Atilde(1,1)>0 && Atilde(2,1)>0 && Atilde(3,1)<0
            s1is=100;
elseif Atilde(1,2)>0 && Atilde(2,2)>0 && Atilde(3,2)<0
            s2is=100;
elseif Atilde(1,3)>0 && Atilde(2,3)>0 && Atilde(3,3)<0
            s3is=100;
elseif Atilde(1,4)>0 && Atilde(2,4)>0 && Atilde(3,4)<0
            s4is=100;
elseif Atilde(1,5)>0 && Atilde(2,5)>0 && Atilde(3,5)<0
            s5is=100;
end
         
% 2. Storage (or speculative) demand shock: +,-,+,+,? && uncertainty impact
% is smaller than uncertainty case
if Atilde(1,1)>0 && Atilde(2,1)<0 && Atilde(3,1)>0 && Atilde(4,1)>0 && Atilde(5,1)<max(Atilde(5,:)) 
           s1is=10;
elseif Atilde(1,2)>0 && Atilde(2,2)<0 && Atilde(3,2)>0 && Atilde(4,2)>0 && Atilde(5,2)<max(Atilde(5,:)) 
           s2is=10;
elseif Atilde(1,3)>0 && Atilde(2,3)<0 && Atilde(3,3)>0 && Atilde(4,3)>0  && Atilde(5,3)<max(Atilde(5,:)) 
           s3is=10;
elseif Atilde(1,4)>0 && Atilde(2,4)<0 && Atilde(3,4)>0 && Atilde(4,4)>0 && Atilde(5,4)<max(Atilde(5,:)) 
           s4is=10;
elseif Atilde(1,5)>0 && Atilde(2,5)<0 && Atilde(3,5)>0 && Atilde(4,5)>0 && Atilde(5,5)<max(Atilde(5,:)) 
           s5is=10;
end
           
% 3. Flow demand shock: +,+,+,?,?
if Atilde(1,1)>0 && Atilde(2,1)>0 && Atilde(3,1)>0 
           s1is=1;
elseif Atilde(1,2)>0 && Atilde(2,2)>0 && Atilde(3,2)>0
           s2is=1;
elseif Atilde(1,3)>0 && Atilde(2,3)>0 && Atilde(3,3)>0 
           s3is=1; 
elseif Atilde(1,4)>0 && Atilde(2,4)>0 && Atilde(3,4)>0 
           s4is=1; 
elseif Atilde(1,5)>0 && Atilde(2,5)>0 && Atilde(3,5)>0 
           s5is=1; 
end  
     
% 4. Uncertainty demand shock: +,-,+,+,+ & larger than speculative shock
if  Atilde(2,1)<0 && Atilde(3,1)>0 && Atilde(4,1)>0 && Atilde(5,1)>0 && Atilde(5,1)==max(Atilde(5,:))  
           s1is=1000;
elseif  Atilde(2,2)<0 && Atilde(3,2)>0 && Atilde(4,2)>0 && Atilde(5,2)>0 && Atilde(5,2)==max(Atilde(5,:))  
           s2is=1000;
elseif  Atilde(2,3)<0 && Atilde(3,3)>0 && Atilde(4,3)>0 && Atilde(5,3)>0 && Atilde(5,3)==max(Atilde(5,:))  
           s3is=1000;
elseif  Atilde(2,4)<0 && Atilde(3,4)>0 && Atilde(4,4)>0 && Atilde(5,4)>0 && Atilde(5,4)==max(Atilde(5,:))  
           s4is=1000;
elseif  Atilde(2,5)<0 && Atilde(3,5)>0 && Atilde(4,5)>0 && Atilde(5,5)>0 && Atilde(5,5)==max(Atilde(5,:))  
           s5is=1000;
end

% Indicator for whether sign restrictions hold
identot=s1is+s2is+s3is+s4is+s5is;

if identot==1111 % signs accepted
    % Rearrange columns of B0inv as needed
               B0inv=zeros(n,n);  
   
              % Shock 1
               if s1is==100 % supply shock
                   B0inv(:,1)=Atilde(:,1);
               elseif s1is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,1);
               elseif s1is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,1);                   
               elseif s1is==0 % residual
                   B0inv(:,4)=Atilde(:,1);
               elseif s1is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,1);
               end

               % Shock 2
               if s2is==100 % supply shock
                   B0inv(:,1)=Atilde(:,2);
               elseif s2is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,2);
               elseif s2is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,2);    
               elseif s2is==0 % residual
                   B0inv(:,4)=Atilde(:,2);
               elseif s2is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,2);
               end

               % Shock 3
               if s3is==100 % supply shock
                   B0inv(:,1)=Atilde(:,3);
               elseif s3is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,3);
               elseif s3is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,3);    
               elseif s3is==0 % residual
                   B0inv(:,4)=Atilde(:,3);
               elseif s3is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,3);
               end

               % Shock 4
               if s4is==100 % supply shock
                   B0inv(:,1)=Atilde(:,4);
               elseif s4is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,4);
               elseif s4is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,4);
               elseif s4is==0 % residual
                   B0inv(:,4)=Atilde(:,4);                   
               elseif s4is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,4);
               end

               % Shock 5
               if s5is==100 % supply shock
                   B0inv(:,1)=Atilde(:,5);
               elseif s5is==1 % flow demand shock
                   B0inv(:,2)=Atilde(:,5);
               elseif s5is==10 % storage demand shock
                   B0inv(:,3)=Atilde(:,5);
               elseif s5is==0 % residual
                   B0inv(:,4)=Atilde(:,5);                   
               elseif s5is==1000 % uncertainty
                   B0inv(:,5)=Atilde(:,5);
               end    
end
end
