# Coordinate descent for nonseparable bridge penalty

Here is the Python and Matlab implementation of high dimensional sparse linear regression with nonseparable bridge penalty based on our paper https://arxiv.org/pdf/2108.03464.pdf.



## Model setting

Consider a penalized linear regression with nonseparable bridge penalty such that


$$
\arg \min _{\beta} \frac{1}{2}||Y-X\beta||^{2}+(2^{\gamma}P+0.5)\log\left(\sum_{j=1}^{P}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)
$$

where $X$ is a $N \times P$ matrix of covariates, $\beta \in \mathbb{R}^{P}$ is assumed to be a sparse vector, and $Y \in \mathbb{R}^N$ is an $N$-vector of response observations.  $b$ and $\gamma$ is the hyperparameter, which is set to $b =C \frac{\log(P)}{P}$. By default $C=2$ and  $\gamma=3$. Most of the time, this CD algorithm is cross-validation free. The default  setting is good enough. If you really want to set $C$ and $\gamma$  by yourself,  you can consider the range of both $C$ and $\gamma$ within 1, 2 and 3. The cross-validation free property make our algorithm quite efficience for large scale problem (e.g $N=2000$, $P=20000$ and $s_{0}=1$).

If the problem is pretty challenging (i.e when the sample size is too small or the noise of the data is large),  then we can use the forward variable screening with cross-validation.




## Algorithm

![WechatIMG76](https://user-images.githubusercontent.com/128662706/230061449-f5dcf61a-b224-49b0-a263-bb13bab6b893.png)

where $\epsilon$ is some suitable small error tolerance, $T$ is the terminal number for the fixed point iteration,  $b$ is the value of the hyper-parameter chosen by the user,   $\beta_{-j}^{i}=(\beta_{1}^{(i+1)},...,\beta_{j-1}^{(i+1)},\beta_{j+1}^{(i)},...,\beta_{p}^{(i)})$  and


$$
u(\beta_{-j})=2\left({\frac{C_{1}(X_{j}^{T}X_{j})^{-1}}{2C_{2}+2[(X_{j}^{T}X_{j})^{-1}|z_{j}|]^{\frac{1}{2^{\gamma}}}}}\right)^{\frac{1}{2-\frac{1}{2^{\gamma}}}}
$$



## Usage

```
from CD import CD_non_separable
from Forward_CV import Forward_CV

beta_estimator,sparsity,sigma2_estimator=CD_non_separable(Y,X,C,s)
beta_estimator_cv,sparsity_cv,sigma2_estimator_cv=Forward_CV(Y,X,gamma,k)
```

1. $Y$ is the vector of response with length $N$. 

2. $X$ is $N \times P$ covariate matrix. 

3. C control the size of hyper-parameter b as we set  $b =C \frac{\log(P)}{P}$. The default value of C is 2.  

4. Both s and gamma are value of $\gamma$. The default value of them are 3. 

5. k is the number of fold for cross-validation. The default value of k is 5




## Reference

```
@article{ke2021bayesian,
  title={Bayesian $ L_\frac{1}{2}$ regression},
  author={Ke, Xiongwen and Fan, Yanan},
  journal={arXiv preprint arXiv:2108.03464},
  year={2021}
}
```
