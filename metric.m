function[L2,L1,sparsity,Ham,FDR,FNDR]=metric(beta,betaTrue)

s=size(betaTrue);
nonzeros=(betaTrue~=0);
zeros=(betaTrue==0);


L2=norm(beta-betaTrue);
L1=norm(beta-betaTrue,1);

nonzero_location=(beta~=0);    
zero_location=1-nonzero_location;
sparsity=sum(nonzero_location);


Ham=sum(nonzeros~=nonzero_location);

FDR=(sum(nonzero_location)-sum(nonzeros.*nonzero_location))/sum(nonzero_location);
FNDR=(sum(zero_location)-sum(zeros.*zero_location))/sum(zero_location);

end