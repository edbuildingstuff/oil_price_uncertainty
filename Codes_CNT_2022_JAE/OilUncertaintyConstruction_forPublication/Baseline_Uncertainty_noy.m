load ferrorsoil_baseline;
svf = load('Oilsvfmeans_baseline.txt');
svy = load('Oilsvymeans_baseline_noy.txt');

% Compute objects from predictors
h   = 1;
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
    yb  = sparse(ybetas_noy(i,:));
    thy = [svy(1,i).*(1-svy(2,i));svy(2,i);svy(3,i).^2];
    xy  = svy(4:end-3,i);
    ut(:,i,:) = compute_uy(xy,thy,yb,py,evf,phif);
    fprintf('Series %d, Elapsed Time = %0.4f \n',i,toc);
end

opu_baseline_noy(:,:) = sqrt(ut(:,1,:));

save opu_baseline_noy opu_baseline_noy

