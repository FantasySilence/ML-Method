# ML-Method
学习机器学习算法时为了理解算法手写的玩意儿

### 目录
- 线性回归[src/algorithm/regression](src/algorithm/regression)
- 逻辑回归[src/algorithm/logistic_regression](src/algorithm/logistic_regression)
- 决策树[src/algorithm/decisiontree](src/algorithm/decisiontree)
- KNN[src/algorithm/KNN](src/algorithm/KNN)
- SVM[src/algorithm/SVM](src/algorithm/SVM)
- LDA[src/algorithm/LDA](src/algorithm/LDA)
- 朴素贝叶斯[src/algorithm/Bayes](src/algorithm/Bayes)
- 集成学习
    - AdaBoost 和 BoostingTree提升树[src/algorithm/ensemble/boost](src/algorithm/ensemble/boost)
    - GradientBoost梯度提升[src/algorithm/ensemble/gradientBoost](src/algorithm/ensemble/gradientBoost)
    - Bagging[src/algorithm/ensemble/bagging](src/algorithm/ensemble/bagging)
    - 随机森林RandomForest[src/algorithm/ensemble/randomForest](src/algorithm/ensemble/randomForest)

## 算法介绍

### 1.线性模型

#### 1.1.一元线性回归: 

$y_{i} = f(x_i) = \theta_{0} + \theta _{1}x_i + \varepsilon _{i} $，假设对于每个$x$都有$y\sim N(\theta_{0} + \theta _{1}x_i,\sigma ^{2} )$，$\theta_{0}, \theta_{1}, \sigma$都是不依赖于$x$的位置参数

- 模型中， $y_i$是$x_i$的线性函数（部分）加上误差项，线性部分反映了由于$x$的变化而引起的$y$的变化。
- 误差项$\varepsilon _{i}$是随机变量：反映了除$x$和$y$之间的线性关系之外的随机因素对$y$的影响，是不能由$x$和$y$之间的线性关系所解释的变异性。误差$\varepsilon _{i} (1\le \varepsilon _{i} \le m)$是独立同分布的，服从均值为0、方差为某定值$\sigma^2$的高斯分布(正态分布)。
- $\theta_{0},\theta_{1}$称为模型的参数

------

MSE(Mean Square Error)均方误差最小化：
$$
(\theta _{0}^{*}, \theta _{1}^{*}) = \underset{(\theta _{0}, \theta _{1})}{argmin} \sum_{i=1}^{m}(f(x_i)-y_i)^{2}=\underset{(\theta _{0}, \theta _{1})}{argmin} \sum_{i=1}^{m}(y_i-\theta _{1}x_i -\theta _{0})^{2}   
$$
基于均方误差最小化来进行模型求解的方法称为“最小二乘法”(least square method)，求解$\theta_0$和$\theta_1$最小化 的过程，称为线性回归模型最小二乘“参数估计” (parameter estimation)。在一元线性回归中最小二乘法就是试图找到一条直线，使所有样本到直线上的欧式距离之和最小。(如下图所示)

![](D:\GitsProject\ML-Method\README_fig\Fig_1.1.png)

记$E_{(\theta _{0},\theta _{1})}=\sum_{i=1}^{m}(y_i-\theta_1x_i-\theta_0)^{2} $，$E_{(\theta _{0},\theta _{1})}$分别对$\theta_{0}, \theta_1$求一阶偏导，得：
$$
\frac{\partial E_{(\theta _{0},\theta _{1})} }{\partial \theta _{1} }=2\left [ \theta_1\sum_{i=1}^{m}x_{i}^2-\sum_{i=1}^{m}(y_i-\theta_0)x_i \right ]=0,\frac{\partial E_{(\theta _{0},\theta _{1})} }{\partial \theta _{1} }=2\left [m\theta _0 - \sum_{i=1}^{m}(y_i - \theta_1x_i)    \right ] = 0
$$
得到最优解的闭式解：
$$
\hat{\boldsymbol{\theta}}_{1}=\frac{\sum_{i=1}^{m} \boldsymbol{y}_{i}\left(x_{i}-\overline{\boldsymbol{x}}\right)}{\sum_{i=1}^{m} \boldsymbol{x}_{i}^{2}-\frac{1}{m}\left(\sum_{i=1}^{m} \boldsymbol{x}_{i}\right)^{2}}, \hat{\boldsymbol{\theta}}_{0}=\frac{1}{m} \sum_{i=1}^{m}\left(y_{i}-\hat{\boldsymbol{\theta}}_{1} x_{i}\right)
$$

------

#### 1.2.多元线性回归

模型：$f(x_i)=\omega^Tx_i+b$，使得$f(x_i)\cong y_i$ ，构造向量$ \hat{\omega} =  (\omega; b)$，相应的把数据集$D$表示为$m\times (d+1)$的矩阵$\mathbf{\mathit{X}}$其中，每行对应一个示例 该行前$d$个元素对应于示例的$d$个属性值，最后一个元素恒置为1，即：
$$
\mathbf{\mathit{X}} = \begin{pmatrix} x_{11} \quad x_{12}  \cdots   x_{1d}\quad 1 \\ x_{21} \quad x_{22}  \cdots   x_{2d}\quad 1 \\ \vdots \quad \quad   \vdots \quad \ddots \quad \vdots \quad \vdots\\ x_{m1} \quad  x_{m2}  \cdots   x_{md}\quad 1\end{pmatrix}=\begin{pmatrix}  \mathbf{\mathit{X}}_{1}^{T}\quad 1 \\  \mathbf{\mathit{X}}_{2}^{T}\quad 1 \\  \vdots \quad \vdots  \\  \mathbf{\mathit{X}}_{m}^{T}\quad 1\end{pmatrix}，y=\begin{pmatrix} y_1\\ y_2\\ \vdots\\ y_3\end{pmatrix}
$$
则$\hat{\omega^*} = \underset{\hat{\omega}}{argmin} (y-\mathbf{\mathit{X}}\hat{\omega})^T(y-\mathbf{\mathit{X}}\hat{\omega})$，令$E_{\hat{\omega}}=(y-\mathbf{\mathit{X}}\hat{\omega})^T(y-\mathbf{\mathit{X}}\hat{\omega})$，对$\hat{\omega}$求导得$\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}} =2X^T(X^T\hat{\omega}-y)$，令上述为零，可求得$\hat{\omega}$最优解的闭式解。分情况讨论：

- 当$X^TX$为满秩矩阵或者正定矩阵时，可得$\hat{\omega ^{*}}= (X^TX)^{-1}X^Ty $，令$\hat{x_i}=(x_i,1)$则得到最终的多元线性回归模型为$f(\hat{x_i})=\hat{x_i}(X^TX)^{-1}X^Ty$ 
- 在现实任务中$X^TX$往往不满秩，此时可以解出多个$\hat{\omega}$，他们都能使得均方误差最小化，选择哪一个解作为输出， 将由学习算法归纳偏好决定，常见的做法是引入正则化(regularization)项。

### 2.决策树

### 3.K近邻

### 4.支持向量机

### 5.贝叶斯分类器 

### 6.集成学习

## 技术栈

Python

## License

GPL-3.0 license