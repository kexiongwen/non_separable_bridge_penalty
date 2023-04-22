function[Y,X,BetaTrue]=data_generator2(n,p,SigmaTrue,rho)

BetaTrue = zeros(p,1);
Beta=zeros(10,1);
Beta(1)=3;
Beta(2)=1.5;
Beta(3)=2;
Beta(4)=1;
Beta(5)=1;
Beta(6)=0.5;
Beta(7)=-0.5;
Beta(8)=2.0;
Beta(9)=-1.2;
Beta(10)=-1;
%Beta(11)=3;
%Beta(12)=1.5;
%Beta(13)=2;
%Beta(14)=1;
%Beta(15)=1;
%Beta(16)=0.5;
%Beta(17)=-0.5;
%Beta(18)=2.0;
%Beta(19)=-1.2;
%Beta(20)=-1;
non_zeros=Beta(randperm(10));
BetaTrue(randsample(p,10))=non_zeros;
Corr=rho.^toeplitz((0:p-1));
X=mvnrnd(zeros(1,p),Corr,n);
Y=X*BetaTrue+SigmaTrue.*randn([n 1]);


end