function ir=IRvar(A,B0inv,n,p,h)

J=[eye(n) zeros(n,(p-1)*n)];
ir=[];
for h=0:h
    ir=cat(3,ir,(J*(A^h)*J')*B0inv);  %cat concatenates arrays: cat(dimension,A,B)
end
