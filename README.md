# Coordinate descent for nonseparable bridge penalty

Here is the Python and Matlab implementation of high dimensional sparse linear regression with nonseparable bridge penalty based on our paper https://arxiv.org/pdf/2108.03464.pdf.



## Model setting

Consider a penalized linear regression with nonseparable bridge penalty such that


$$
\arg \min _{\beta} \frac{1}{2}||Y-X\beta||^{2}+(2^{\gamma}p+0.5)\log\left(\sum_{j=1}^{P}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)
$$

where $X$ is a $N \times P$ matrix of covariates, $\beta \in \mathbb{R}^{P}$ is assumed to be a sparse vector, and $Y \in \mathbb{R}^N$ is an $N$-vector of response observations.  $b$ is the hyperparameter, which is set to $b \propto \frac{\log(P)}{P}$. $\gamma$ is set to $1$ or $2$.




## Algorithm

![WechatIMG76](https://user-images.githubusercontent.com/128662706/230061449-f5dcf61a-b224-49b0-a263-bb13bab6b893.png)

where $\epsilon$ is some suitable small error tolerance, $T$ is the terminal number for the fixed point iteration,  $b$ is the value of the hyper-parameter chosen by the user,   $\beta_{-j}^{i}=(\beta_{1}^{(i+1)},...,\beta_{j-1}^{(i+1)},\beta_{j+1}^{(i)},...,\beta_{p}^{(i)})$  and


$$
u(\beta_{-j})=2\left({\frac{C_{1}(X_{j}^{T}X_{j})^{-1}}{2C_{2}+2[(X_{j}^{T}X_{j})^{-1}|z_{j}|]^{\frac{1}{2^{\gamma}}}}}\right)^{\frac{1}{2-\frac{1}{2^{\gamma}}}}
$$






## Reference

```
@article{ke2021bayesian,
  title={Bayesian $ L_\frac{1}{2}$ regression},
  author={Ke, Xiongwen and Fan, Yanan},
  journal={arXiv preprint arXiv:2108.03464},
  year={2021}
}
```
