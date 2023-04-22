function[L2,L1,sparsity,Ham,FDR,FNDR,sigma2,L2_std,L1_std,sparsity_std,Ham_std,sigma2_std]=Experiments(repeat,n,p,sigma,rho)

L2_m=ones(repeat,1);
L1_m=ones(repeat,1);
sparsity_m=ones(repeat,1);
Ham_m=ones(repeat,1);
FDR_m=ones(repeat,1);
FNDR_m=ones(repeat,1);
sigma2_m=ones(repeat,1);

for i=1:repeat

    [Y,X,BetaTrue]=data_generator2(n,p,sigma,rho);
    [beta,~,sigma2]=CD_NSB(Y,X,2,3);
    [L2,L1,sparsity,Ham,FDR,FNDR]=metric(beta,BetaTrue);

    L2_m(i)=L2;
    L1_m(i)=L1;
    sparsity_m(i)=sparsity;
    Ham_m(i)=Ham;
    FDR_m(i)=FDR;
    FNDR_m(i)=FNDR;
    sigma2_m(i)=sigma2;
   

end

L2=mean(L2_m);
L2_std=std(L2_m);
L1=mean(L1_m);
L1_std=std(L1_m);
sparsity=mean(sparsity_m);
sparsity_std=std(sparsity_m);
Ham=mean(Ham_m);
Ham_std=std(Ham_m);
FDR=mean(FDR_m);
FNDR=mean(FNDR_m);
sigma2=mean(sigma2_m);
sigma2_std=std(sigma2_m);

end