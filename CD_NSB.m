function[beta,sparsity,sigma2]=CD_NSB(Y,X,C,s)

XTX=X'*X;
S=size(X);
b=C*log(S(2))/S(2);
beta=zeros(S(2),1);
beta_previous=ones(S(2),1);
Z=ones(S(2),1);    
a=0.5;
C1=(S(2)+a/(2.^s));
iteration=1;
power=1/(2-0.5.^s);

ink1=X(:,2:end)*beta(2:end,:);
ink2=zeros(S(1),1);

while (norm(beta-beta_previous)>1e-3) && (iteration<20)

    beta_previous=beta;
    iteration=iteration+1;

    for j=1:S(2)

        if (j~=1) && (j~=S(2))
                Z(j)=X(:,j)'*(Y-ink1-ink2)/XTX(j,j);
        elseif j==1
                Z(1)=X(:,1)'*(Y-ink1)/XTX(1,1);
        else
                Z(S(2))=X(:,S(2))'*(Y-ink2)/XTX(S(2),S(2));
        end

        C2=1/b+sum(abs(beta).^(0.5.^s))-abs(beta(j,:)).^(0.5.^s);
        
        if abs(Z(j))<2*((C1/(C2+abs(Z(j)).^(0.5.^s)))*0.5/XTX(j,j)).^power
            beta(j,:)=0;
        else
            
            beta_old=abs(Z(j));
            beta_new=1000;
            k=1;
            while abs(beta_old-beta_new)>1e-4 && (k<=20) && (beta_new>=0)
                beta_old=beta_new;
                beta_new=abs(Z(j))-C1/XTX(j,j)/(beta_old+C2*beta_old.^(1-0.5.^s));
                k=k+1;
            end

            if k>20 || beta_new<0
                beta(j,:)=0; 
            else
                beta(j,:)=beta_new*sign(Z(j));
            end
        end

        if (j~=1) && (j~=S(2))
                ink1=ink1-X(:,j+1)*beta(j+1,:);
                ink2=ink2+X(:,j)*beta(j,:);
        elseif j==1
                ink1=ink1-X(:,2)*beta(2,:);
                ink2=X(:,1)*beta(1,:);
        else
                ink1=ink2+X(:,j)*beta(j,:)-X(:,1)*beta(1,:);
                ink2=zeros(S(1),1);
        end

    end

    sparsity=nnz(beta);
    rrs=Y-X*beta;
    sigma2=rrs'*rrs/(S(1)-sparsity);

end