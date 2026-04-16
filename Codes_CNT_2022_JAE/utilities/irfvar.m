% IRFVAR.M


function [IRF]=irfvar(A,Atilde,p,n,H)

J=[eye(n,n) zeros(n,n*(p-1))];
IRF=reshape(J*A^0*J'*Atilde,n^2,1);
for i=1:H
	IRF=([IRF reshape(J*A^i*J'*Atilde,n^2,1)]);
end;


