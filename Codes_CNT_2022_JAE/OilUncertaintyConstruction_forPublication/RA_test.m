function [Result,A,Ainf,Asup] = RA_test(x,method)
% [Result,A,Ainf,Asup] = RA_test(x,method) carry out the reverse
% arrangement test [1] to detect if a signal can be considered as
% stationary or not (trend detection).
%% Syntax
%
%  Result = RA_test(x)
% [Result,A,Ainf,Asup] = RA_test(x,2)
% [Result,A,Ainf,Asup] = RA_test(x,1)
%
% Input:
%   * x: vector: [1 x N] or [N x 1] time series analysed:
%   * method: 1 or 2
%       - 1: bendat and piersol [1, page 99]
%       - 2: Siegel and Castellan,1988 [2]
%
% Output
%   * Result: boolean: 1 if data is stationary, or 0  otherwise
%   * A: scalar: total number of reverse arrangements
%   * Ainf: scalar: lower boundary
%   * Asup: scalar: upper boundary
%% Author info
%  Etienne Cheynet, University of Stavanger, Norway - 28/05/2016
%% References
% [1] Bendat, J. S., and A. G. Piersol. "Random Data-Analysis and Measurement Procedures." (1989): 354-355.
% [2] Siegel, Sidney, and N. J. Castellan. "Non-para-metric statistics for the behavioral sciences." (1988).

%%
narginchk(1,2);
if nargin==1,    method = 2;end
if min(size(x))>1, error('x must be a vector, not a matrix !'); end
x = x(:); % force the input to be a column vector
N = numel(x);


% Calculation of h  [1, page 97]
h = zeros(N);
count = 0;
for ii= 1:N,
    for jj=1:N,
        while ii<jj,
            if x(ii)>x(jj)
                h(ii,jj) = 1;
            else
                h(ii,jj) = 0;
            end
            count = count+1;
            if count > N,
                break
                fprintf('loop "while" failed \n');
            end
        end
    end
end
% Calculation of A (page 97)
A = zeros(1,N-1);
for ii= 1:N,
    A(ii)= sum(h(ii,ii+1:end));
end
A = sum(A(1:N-1));

meanA = N*(N-1)/4; % mean value
stdA = sqrt(N*(2*N+5)*(N-1)./72);% standard deviation
z = 1.96; % 95% confidence level

%% Apply method 1 or 2
switch method
    case 1 % original method: Bendat and piersol
        % upper boundary (page 94)
        Asup = ceil(z*stdA+meanA-0.5);
        % Lower boundary (page 94)
        Ainf = floor(meanA-z*stdA+0.5);
        % Check the test
        if and(A<Asup,A>Ainf)
            Result = 1; % pass the test -> signal is stationnary
        else
            Result = 0;% fail the test -> signal is  NOT stationnary
        end
    case 2 % alternativ: Siegel and Castellan,1988
        A = (A-meanA)/stdA;
        Asup = z;
        Ainf = -z;
        if abs(A)<z,
            Result = 1; % pass the test -> signal is stationnary
        else
            Result = 0;% fail the test -> signal is  NOT stationnary
        end
    otherwise
        error(' method is unknown \n')
end
end

