function [R2,mR2,mR2_F,R2_T,tntop_s,tntop_mR2] = mrsq(Fhat,lamhat,ve2,series,ntop)
% =========================================================================
% DESCRIPTION
% This function computes the R-squared and marginal R-squared from
% estimated factors and factor loadings.
%
% -------------------------------------------------------------------------
% INPUTS
%           Fhat    = estimated factors (one factor per column)
%           lamhat  = factor loadings (one factor per column)
%           ve2     = eigenvalues of covariance matrix
%           series  = series names
%           ntop    = number of top series that load most that we want to
%           look at
%
% OUTPUTS
%           R2      = R-squared for each series for each factor
%           mR2     = marginal R-squared for each series for each factor
%           mR2_F   = marginal R-squared for each factor
%           R2_T    = total variation explained by all factors
%           tntop_s   = top 10 series that load most heavily on each factor
%           tntop_mR2 = marginal R-squared corresponding to top 10 series
%                     that load most heavily on each factor 
%           
% -------------------------------------------------------------------------
% NOTES
% Authors: Michael W. McCracken and Serena Ng
% Date: 6/7/2017
% Version: MATLAB 2014a
% Required Toolboxes: None
%
% =========================================================================
% FUNCTION

% N = number of series, ic = number of factors
[N,ic] = size(lamhat); 

% Preallocate memory for output 
R2 = NaN(N,ic);                           
mR2 = NaN(N,ic);
tntop_s=cell(ntop,ic);
tntop_mR2=NaN(ntop,ic);

% Compute R-squared and marginal R-squared for each series for each factor
for i = 1:ic
    R2(:,i)  = (var(Fhat(:,1:i)*lamhat(:,1:i)'))';  
    mR2(:,i) = (var(Fhat(:,i)*lamhat(:,i)'))';
end

% Compute marginal R-squared for each factor 
mR2_F = ve2./sum(ve2);
mR2_F = mR2_F(1:ic)';

% Compute total variation explained by all factors
R2_T=sum(mR2_F);

% Sort series by marginal R-squared for each factor
[vals,ind] = sort(mR2,'descend');

% Get top 3 series that load most heavily on each factor and the
% corresponding marginal R-squared values
for i=1:ic
    tntop_s(:,i)=series(ind(1:ntop,i));
    tntop_mR2(:,i)=vals(1:ntop,i);
end

