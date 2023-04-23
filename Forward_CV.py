import numpy as np

def CD_non_separable_CV(Y,X,beta_initial,t,s=3):
    
    XTX=X.T@X
    N,P=np.shape(X)
    if t==0:
        b=10000
    else:
        b=1/t
    beta=beta_initial
    beta_previous=np.ones((P,1))
    Z=np.ones(P)    
    a=0.5
    C1=(P+a/(2**s))
    iteration=1
    power=1/(2-0.5**s)

    ink1=X[:,1:]@beta[1:,:]
    ink2=np.zeros((N,1))
        
    while (np.linalg.norm(beta-beta_previous)>1e-4 and iteration<20):

        beta_previous=np.copy(beta)
        iteration=iteration+1

        for j in range(0,P):

            if (j!=0) & (j!=P-1):
                Z[j]=X[:,j:j+1].T@(Y-ink2-ink1)/XTX[j,j]
            elif j==0:
                Z[0]=X[:,0:1].T@(Y-ink1)/XTX[0,0]
            else:
                Z[P-1]=X[:,P-1:P].T@(Y-ink2)/XTX[P-1,P-1]
            
            C2=1/b+np.sum(np.abs(beta)**(0.5**s))-np.abs(beta[j,:])**(0.5**s)
            
            
            if np.abs(Z[j])<2*((C1/(C2+np.abs(Z[j])**(0.5**s)))*0.5/XTX[j,j])**power:
                beta[j,:]=0
            else:

                beta_old=np.abs(Z[j])
                beta_new=1000
                k=1
                
                while (np.abs(beta_old-beta_new)>1e-4 and k<=20 and beta_new>=0):

                    beta_old=np.copy(beta_new)
                    beta_new=np.abs(Z[j])-C1/XTX[j,j]/(beta_old+C2*beta_old**(1-0.5**s))
                    k=k+1

                if k>20 or beta_new<0:
                    beta[j,:]=0 
                else:
                    beta[j,:]=beta_new*np.sign(Z[j])
            
            if (j!=0) & (j!=P-1):
                ink1=ink1-X[:,j+1:j+2]@beta[j+1:j+2,:]
                ink2=ink2+X[:,j:j+1]@beta[j:j+1,:]
            elif j==0:
                ink1=ink1-X[:,1:2]@beta[1:2,:]
                ink2=X[:,0:1]@beta[0:1,:]
            else:
                ink1=ink2+X[:,j:j+1]@beta[j:j+1,:]-X[:,0:1]@beta[0:1,:]
                ink2=np.zeros((N,1))

    sparsity=np.count_nonzero(beta!=0)
    sigma2_estimator=(Y-X@beta).T@(Y-X@beta)/(N-sparsity)
            
    result=(beta,sparsity,sigma2_estimator)
    return result


def Forward_CV(Y,X,gamma=3,k=5):
    
    _,P=np.shape(X)
    
    data=np.concatenate((Y,X),axis=1)
    np.random.shuffle(data)
    chunks=np.split(data,k)
    step=10
    grid=np.linspace(0,step*k,step)
    validation_error=np.zeros((k,step))
    validation_error_sum=np.zeros(step)
    
    
    for j in range(0,k):
        
        beta_path=np.zeros((P,step+1))
        
        Y_valid=chunks[j][:,0:1]
        X_valid=chunks[j][:,1:]
        
        chunks1=chunks[:]
        chunks1.pop(j)
        
        data_train=np.concatenate(chunks1,axis=0)
        
        Y_train=data_train[:,0:1]
        X_train=data_train[:,1:]
        
    
        for i in range(1,step+1):

            beta_path[:,i:i+1],_,_=CD_non_separable_CV(Y_train,X_train,beta_initial=beta_path[:,i-1:i],t=grid[i-1],s=gamma)
            
            validation_error[j,i-1]=np.linalg.norm(Y_valid-X_valid@beta_path[:,i:i+1],2)
                 
    validation_error_sum=np.sum(validation_error,axis=0)
            
    location=np.argmin(validation_error_sum)
    
    beta_estimator=np.zeros((P,1))
    
    for j in range(0,location+1):
    
        beta_estimator,sparsity,sigma2_estimator=CD_non_separable_CV(Y,X,beta_initial=beta_estimator,t=grid[j],s=gamma)
    
 
    result=(beta_estimator,sparsity,sigma2_estimator)
    return result    