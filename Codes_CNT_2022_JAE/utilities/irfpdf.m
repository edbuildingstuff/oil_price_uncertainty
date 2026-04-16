function f = irfpdf(Atilde,EvecB,NT,nuT,ST,vecB)

% Purpose:
% This code computes the log of the joint posterior density of structural impulse responses up to constant
% input :
% Atilde: Obtained from rotating matrix A by matrix U, i.e., Atilde=A*U
% EvecB : Posterior mean of vec(B)
% NT    : N0+X'*X
% nuT   : degrees of freedom for the inverse-Wishart distribution
% ST    : Scale matrix for the inverse-Wishart distribution
% vecB  : Posterior draw of vec(B) = vec([c Phi1 ... Phip]')
% output:
% y     : Log of the joint posterior density up to constant


n = size(Atilde,1);           % The number of variables
p = (size(vecB,1)/n-1)/n; % The order of the VAR model
index = kron(ones(n,1),[0;ones(n*p,1)]);


% Communication Matrix
Kn=[];
for i=1:n
    for j=1:n
        E      = zeros(n,n);
        E(i,j) = 1;
        Kn     = [Kn reshape(E,n*n,1)];
    end
end

% Duplication Matrix
Dn = [];
for j=1:n
    for i=j:n
        E      = zeros(n,n);
        E(i,j) = 1;
        E(j,i) = 1;
        Dn     = [Dn reshape(E,n*n,1)];
    end
end
Dnplus = (Dn'*Dn)\(Dn');

% Duplication Matrix" for Non-Symmetric Matrices (such as A)
Dbarn = [];
for j=1:n
    for i=j:n
        E      = zeros(n,n);
        E(i,j) = 1;
        Dbarn     = [Dbarn reshape(E,n*n,1)];
    end
end

% En Matrices
En = [];
for j=2:n
    for i=1:j-1
        E      = zeros(n,n);
        E(i,j) = 1;
        E(j,i) = -1;
        En     = [En reshape(E,n*n,1)];
    end
end
Enplus = (En'*En)\(En');

% Compute reduced-form impulse responses
B     = (reshape(vecB,1+n*p,n))';
Phi   = [B(:,1+1:end);eye(n*(p-1)) zeros(n*(p-1),n)];
Theta = eye(n);
for i = 1:p
    Phii  = Phi^i;
    Theta = [Theta Phii(1:n,1:n)];
end

% Compute the Jacobian of vec(Theta) 
Sigma       = Atilde*Atilde';
A           = chol(Sigma)';
U           = (A)\Atilde;
v=eig(U);
j=1;
while j<=n
     if (abs(real(v(j))+1)<sqrt(eps))&(abs(imag(v(j))<sqrt(eps)))
         W=diag([-1; ones(n-1,1)]);
         A=A*W;
         U=W*U;
         %Atilde=(A*W)*(W*U)=A*U is unchanged
         j=n+1; % Terminate while statement
     else
         j=j+1;
     end;    
end;  
S           = eye(n)-2*((eye(n)+U)\eye(n));
s           = Enplus*reshape(S,n*n,1);
Ju          = kron(inv(eye(n)-S)',(eye(n)-S)\eye(n))*En; 
J1          = [kron(eye(n),A)*Ju kron(U',eye(n))*Dbarn];
J3          = Dnplus*(kron(A,eye(n))+kron(eye(n),A)*Kn)*Dbarn;

logdetJ     = -log(abs(det(J1)))-0.5*n*p*log(det(Sigma))+log(abs(det(J3))); 
VvecB       = kron(Sigma,NT\speye(n*p+1));
VvecB = (VvecB + VvecB')./2; % eliminate numerical precision errors
VvecBinv    = VvecB(index==1,index==1)\speye(n*(n*p));
logdetVvecB = logdet(VvecB(index==1,index==1));
f           = logdetJ-0.5*logdetVvecB-0.5*(vecB(index==1,1)-EvecB(index==1,1))'*VvecBinv*(vecB(index==1,1)-EvecB(index==1,1));
f           = f-0.5*(nuT+n+1)*log(det(Sigma))-0.5*trace((nuT*ST)/(Sigma));
f           = f-(n-1)*log(det(eye(n)+S));

