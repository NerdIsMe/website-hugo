---
title: "Logistic Regression"
date: 2021-03-26T19:19:27+08:00
draft: False
toc:
  enable: true
  auto: true
---
幫資料分群，找出一條regression，兩側的資料**不同屬性**

## Input data & Functions
### Independent & Dependent Variables
對於一個資料點$(x_i, t_i)$：
- 有 $m$個 Independent variable 的資料：$x_i \in R^m$
- dependent variables: 屬於 $C_1:t_i = 1$ or $C_2: t_i = 0$

### Predictions
- $\hat y_i = f(\phi (x_i)), 0<\hat y_i < 1$
    - $\hat y_i$ 是資料點 $x_i$ 屬於 $C_1$ 的機率

### Probability Function: Sigmodal Function
 給定一個 $z$，屬於$C_1$的機率為：
 $$\sigma(z) = \frac{1}{1+e^{-z}}$$
 ![](https://i.imgur.com/7scZddK.png =60%x)

#### 將 independent variable 轉成 $z$ 放入 Sigmodal function:
- ![](https://i.imgur.com/mi8Fqdj.png =60%x)
- $\sigma(z)$ 就是 給定權重$w$, bias $b$，在 $x_i$ 的情況下，$C_1$ 發生的機率 $\implies P(C_1|\phi(x_i)) = y(\phi) = \sigma(w^T\phi)$

## Parameter Estimation
### Posterior distribution
- Posterior distribution: 
$$P(w|t) = \frac{P(t,w)}{P(t)} = \frac{P(t|w)P(w)}{P(t)} \propto P(t|w)P(w)$$
    - 後驗機率：給定已知 $t$，找 $w$ 的機率
- $\implies$ Posterior $\propto$ <font color = blue>Likelyhood</font> * <font color = red>Prior</font>


### Likelyhood Function
- 給定 parameter $w$，==我有多少機率求得 $t$==
- Likelyhood function $P(t|w)$: 
$$P(t|w) = \Pi_{n = 1}^N\ \{y_n^{t_n}(1-y_n)^{1-t_n}\}$$
    - $y_n = \sigma (w^T\phi(x_n)), t = (t_1, t_2,..,t_N)^T$
    - 如果$t_n = 1$，機率為 $y_n^1(1-y_n)^0 = y_n$
    如果$t_n = 0$，機率為 $y_n^0(1-y_n)^1 = 1-y_n$
    
- 對 liklyhood function 取 log:
    $$\begin{split}ln\ P(t|w) = \Sigma_{n = 1}^N\ t_n\ ln\ y_n + (1-t_n)ln(1-y_n)\end{split}$$
### Prior: 先驗機率
- 還沒觀察到數據之前，對於模型的機率**有一些估計或了解**的概率分布。
- 觀察到數據之後所更新的模型機率，為後驗機率。
- 基本上，所有的機率分布都能選來作為 $p$ 不確定性的概論分布
    但 Gaussian prior 一班來說，會讓 posterior 計算上比較方便
    
#### Gaussian Prior
- 在這裡我們使用 Gaussian prior $p(w) = N(w|m_0, s_0)$
    - $m_0$ 是給定的均數, $s_0$ 是variance，兩者都是hyperparameter
    
- $\begin{split} p(w) =N(w|m_0, s_0) = \frac{exp(-\frac{1}{2} (w-m_0)^Ts^{-1}_0(w-m_0))}{\sqrt{(2 \pi)^k|s_0|}}\end{split}$

 對 $p(w)$ 取 $ln$
  $$\begin{split}ln\ p(w) &= -\frac{1}{2} (w-m_0)^Ts^{-1}_0(w-m_0)-\frac{k}{2}ln2\pi- \frac{1}{2}lns_0
 \\ &= -\frac{1}{2} (w-m_0)^Ts^{-1}_0(w-m_0) + \text{constant}
 \end{split}$$


若假設 $m_0 = 0, s_0 = \frac{1}{\lambda}$:
 $$ln\ p(w)= -\frac{\lambda}{2}w^Tw \\
    \implies L2\ Regularization$$
- 某種程度上，假設 $m_0 =0$ 是很合理的
    1. 有助於 gaussian prior 的計算
    2. 對於沒有影響力的變數，權重 = 0 

### Evaluation of Posterior Distribution
- Posterior distribution: $P(w|t) \propto P(t|w)P(w)$
$$\implies lnP(w|t) \propto \Sigma_{n = 1}^N\ [t_n\ ln\ y_n + (1-t_n)ln(1-y_n)] -\frac{1}{2}w^Tw$$

$$\text{Error Function} = -\Sigma_{n = 1}^N\ [t_n\ ln\ y_n + (1-t_n)ln(1-y_n)] +\frac{1}{2}w^Tw$$
## Maximize Posterior Distribution

### Newtons Method
Gradient descent 僅僅用一次微分，有時候頗慢。Newton's Method 有用到二次微分，速度通常較快。

#### $R^1$ 的 linear Approximation：
-  $x^k$ 的 Linear approximation: $f_L(x) = f(x^k) + \bigtriangledown f(x^k)(x-x^k)$

- $f'(x_n) = \frac{\bigtriangledown y}{\bigtriangledown x} = \frac{f(x_n) -0}{x_n-x_{n+1}}\\
\implies x_{n+1} =x_n -\frac{f(n)}{f'(x_n)}$

如果今天想要找 $f(x) = 0$ 的解：
- $f_L(x^{k+1}) = f(x^k) + \bigtriangledown f(x^k)(x-x^k) =0$, 一直重複這個步驟，可以逼近 正解
- ![](https://i.imgur.com/7Td3P1K.png =40%x) 從$x_0$ 移動到 $x_1$，最終會逼近到 $\overline x$

#### $R^2$ 的 linear Approximation：求 Minimum or Maximum
- minimum 是一階微分 $\bigtriangledown f(x)$，求$\bigtriangledown f(x) = 0$
    對 gradient 再使用一次 Newton:
    $$x_{n+1} = x_n -\frac{\bigtriangledown f(x_n)}{\bigtriangledown^2 f(x_n)}$$
- 如果 $x$ 是一個矩陣，則會變為：
    $$x_{n+1} = x_n-H^{-1}\bigtriangledown f(x_n)$$
    - $H = \bigtriangledown^2 f(x_n) = \frac{f(x_n)}{\partial x_n^T\partial x_n}$
    - ![](https://i.imgur.com/nC2R6iW.png =60%x)


### Newton Method to minimize error function
$$\text{Error Function = -Likelyhood Function} =- P(t|w) = -\Sigma_{n = 1}^N\ t_n\ ln\ y_n + (1-t_n)ln(1-y_n)+\frac{\lambda}{2}w^Tw$$

#### Gradient of error function
$$\begin{split}\bigtriangledown E(w) =\frac{E(w)}{\partial w} &= \frac{-\Sigma_{n = 1}^N\ t_n\ ln\ y_n + (1-t_n)ln(1-y_n)+\frac{\lambda}{2}w^Tw}{\partial w}\\
&=-\Sigma_{n = 1}^N(t_n(1-\sigma(w^T\phi_n))\phi_n + (t_n - 1)\sigma(w^T\phi_n)\phi_n) + \lambda w\\
&= -\Sigma_{n = 1}^N (t_n -\sigma(w^T\phi_n))+ \lambda w\\
& = \Sigma_{n = 1}^N (\sigma(w^T\phi_n) - t_n)+ \lambda w \\
& = \Phi^T(y-t) + \lambda w
\end{split}$$
- ![](https://i.imgur.com/SwXxO0K.png =40%x)
- $\frac{\partial y}{\partial w} = \frac{\partial (w^T\phi_n)}{\partial w} = \phi_n$
- $\frac{y}{\partial w} =\frac{\partial \sigma(w^T\phi_n)}{\partial w} = \frac{\partial \sigma(w^T\phi_n)}{\partial (w^T\phi_n)}\frac{\partial w^T\phi_n}{\partial w} = \sigma (w^T\phi_n)(1-\sigma(w^T\phi_n))\phi_n$

#### Hessian of the error function

$$\begin{split}\bigtriangledown \bigtriangledown E(w) = \frac{\bigtriangledown E(w)}{\partial w^T} &= \Sigma_{n = 1}^N y_n(1-y_n)\phi_n\phi_n^T+\lambda I\\
&=\Phi^TR\Phi+\lambda I
\end{split}$$
- $R$ is a $N\times N$ diangonal matrix with elements $R_{n\times n}= y_n(1-y_n)$, 

#### Applying newton's method on logistic regression
1. Get Error function $E(w)$
2. Compute gradient $\bigtriangledown E(w)$
3. Compute hessian $H(w) = \bigtriangledown^2 E(w)$
4. Newton's method: $w^{n+1} = w^n - H(w_n)^{-1}\bigtriangledown E(w_n)$
  