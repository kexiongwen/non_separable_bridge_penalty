tic
n=2000;
p=20000;
BetaTrue = zeros(p,1);
BetaTrue(1)=3;
BetaTrue(2)=1.5;
BetaTrue(5)=2;
BetaTrue(10)=1;
BetaTrue(13)=1;
BetaTrue(19)=0.5;
BetaTrue(26)=-0.5;
BetaTrue(31)=2.0;
BetaTrue(46)=-1.2;
BetaTrue(51)=-1;
SigmaTrue=3.^0.5;
Corr=0.5.^toeplitz((0:p-1));
X=mvnrnd(zeros(1,p),Corr,n);
Y=X*BetaTrue+SigmaTrue.*randn([n 1]);
toc

C=2;
s=3;


tic
[beta,~,sigma2]=CD_NSB(Y,X,C,s);
toc

[L2,L1,sparsity,Ham,FDR,FNDR]=metric(beta,BetaTrue);


