# 特征选择与变换习题

**姓名：** 甘云冲

**学号：** 2101213081

## 1.

### (1)

$\boldsymbol{H}\boldsymbol{e} = 0$:
$$
\boldsymbol{He} =(\boldsymbol{I}-\boldsymbol{e}\boldsymbol{e}^T)\boldsymbol{e} = \boldsymbol{I}\boldsymbol{e}-\boldsymbol{e}(\boldsymbol{e}^T\boldsymbol{e}) = \boldsymbol{e} - \boldsymbol{e} = \boldsymbol{0}
$$

$\boldsymbol{HH}=\boldsymbol{H}$:


$$
\boldsymbol{HH} = (\boldsymbol{I}-\boldsymbol{e}\boldsymbol{e}^T)(\boldsymbol{I}-\boldsymbol{e}\boldsymbol{e}^T)= \boldsymbol{I} - \boldsymbol{e}\boldsymbol{e}^T-\boldsymbol{e}\boldsymbol{e}^T+\boldsymbol{e}(\boldsymbol{e}^T\boldsymbol{e})\boldsymbol{e}^T =\boldsymbol{I}-\boldsymbol{e}\boldsymbol{e}^T = \boldsymbol{H}
$$

$(\boldsymbol{x}_1-\bar{\boldsymbol{x}},\ldots,
\boldsymbol{x}_n-\bar{\boldsymbol{x}})^T = \boldsymbol{HX}$:
$$
(\bar{\boldsymbol{x}}, \bar{\boldsymbol{x}},\ldots,\bar{\boldsymbol{x}})^T = [\bar{\boldsymbol{x}}(1,1,...,1)]^T = (\boldsymbol{\boldsymbol{X}^T\boldsymbol{e}\boldsymbol{e}^T})^T = \boldsymbol{e}\boldsymbol{e}^T\boldsymbol{X}
$$

$$
(\boldsymbol{x}_1-\bar{\boldsymbol{x}},\ldots,
\boldsymbol{x}_n-\bar{\boldsymbol{x}})^T = \boldsymbol{X}-\boldsymbol{e}\boldsymbol{e}^T\boldsymbol{X} = (\boldsymbol{I}-\boldsymbol{e}\boldsymbol{e}^T)\boldsymbol{X} = \boldsymbol{HX}
$$

$\boldsymbol{\Sigma}=\frac1n \boldsymbol{X}^T\boldsymbol{H}\boldsymbol{X}$:

$$
\boldsymbol{\Sigma} = \frac1 n (\boldsymbol{HX})^T(\boldsymbol{HX}) = \frac{1}{n} \boldsymbol{X}^T(\boldsymbol{H}^T\boldsymbol{H})\boldsymbol{X} = \frac{1}{n} \boldsymbol{X}^T(\boldsymbol{H}\boldsymbol{H})\boldsymbol{X} = \frac{1}{n} \boldsymbol{X}^T\boldsymbol{H}\boldsymbol{X}
$$

### (2)

$\boldsymbol{v}_i$为$\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{X}$的特征值：
$$
\begin{aligned}
\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{X}\boldsymbol{v}_i &= \lambda_i \boldsymbol{v}_i
\\
\boldsymbol{HXX}^T\boldsymbol{H}\boldsymbol{X}\boldsymbol{v}_i &= \lambda_i\boldsymbol{HX}\boldsymbol{v}_i
\\
\boldsymbol{HXX}^T\boldsymbol{H}(\boldsymbol{H}\boldsymbol{X}\boldsymbol{v}_i) &= \lambda_i(\boldsymbol{HX}\boldsymbol{v}_i)
\end{aligned}
$$
$\boldsymbol{u}_i$为$\boldsymbol{HXX}^T\boldsymbol{H}$的特征值：
$$
\begin{aligned}
\boldsymbol{HXX}^T\boldsymbol{H}\boldsymbol{u}_i &= l_i \boldsymbol{u}_i
\\
\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{HXX}^T\boldsymbol{H}\boldsymbol{u}_i &= l_i \boldsymbol{X}^T\boldsymbol{H}\boldsymbol{u}_i
\\
\boldsymbol{X}^T\boldsymbol{HX}(\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{u}_i) &= l_i (\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{u}_i)
\end{aligned}
$$
又由于：
$$
\boldsymbol{v}_i^T\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{X}\boldsymbol{v}_i = (\boldsymbol{H}\boldsymbol{X}\boldsymbol{v}_i)^T(\boldsymbol{H}\boldsymbol{X}\boldsymbol{v}_i) = \lambda _i,\qquad \boldsymbol{u}_i^T\boldsymbol{u}_i = 1
\\
\boldsymbol{u}_i^T\boldsymbol{HXX}^T\boldsymbol{H}\boldsymbol{u}_i = (\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{u}_i)^T(\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{u}_i) = l_i,\qquad\boldsymbol{v}_i^T\boldsymbol{v}_i = 1
$$
可以知道对应的比例为$\sqrt{\lambda_i}$和$\sqrt{l_i}$，即$\boldsymbol{v}_i = \frac{\boldsymbol{X}^T\boldsymbol{H}\boldsymbol{u}_i}{\sqrt{l_i}}, \boldsymbol{u}_i = \frac{\boldsymbol{HXv}_i}{\sqrt{\lambda_i}}$。

## 2.

### (1)

由于$\boldsymbol{u}$为$\boldsymbol{B}$的特征向量：
$$
l_i \boldsymbol{u}_i = \boldsymbol{HXX}^T\boldsymbol{H}\boldsymbol{u}_i = \boldsymbol{HHXX}^T\boldsymbol{H}\boldsymbol{u}_i = l_i\boldsymbol{Hu}_i
\\
\Rightarrow \boldsymbol{u}_i = \boldsymbol{Hu}_i 
$$
于是有：
$$
\boldsymbol{H}\hat{\boldsymbol{X}} = \boldsymbol{H}(\sqrt{l_1}\boldsymbol{u}_1,\ldots,\sqrt{l_n}\boldsymbol{u}_n)
= (\sqrt{l_1}\boldsymbol{H}\boldsymbol{u}_1,\ldots,\sqrt{l_n}\boldsymbol{H}\boldsymbol{u}_n)
= (\sqrt{l_1}\boldsymbol{u}_1,\ldots,\sqrt{l_n}\boldsymbol{u}_n) = \hat{\boldsymbol{X}}
$$


### (2)

$$
\hat{\boldsymbol{B}}=\boldsymbol{H}\hat{\boldsymbol{X}}\hat{\boldsymbol{X}}^T\boldsymbol{H} = \boldsymbol{H}\hat{\boldsymbol{X}}\hat{\boldsymbol{X}}^T\boldsymbol{H}^T = \boldsymbol{H}\hat{\boldsymbol{X}}(\boldsymbol{H}\hat{\boldsymbol{X}})^T = \hat{\boldsymbol{X}}\hat{\boldsymbol{X}}^T = \boldsymbol{B}
$$

### (3)

$$
\begin{aligned}
s_{ij} &= (\boldsymbol{x}_i - \boldsymbol{x}_j)^T(\boldsymbol{x}_i -\boldsymbol{x}_j)
\\
&= ((\boldsymbol{x}_i - \bar{\boldsymbol{x}}) - (\boldsymbol{x}_j - \bar{\boldsymbol{x}}))^T((\boldsymbol{x}_i - \bar{\boldsymbol{x}}) - (\boldsymbol{x}_j - \bar{\boldsymbol{x}}))
\\
&= (\boldsymbol{x}_i - \bar{\boldsymbol{x}})^T(\boldsymbol{x}_i - \bar{\boldsymbol{x}}) + (\boldsymbol{x}_j - \bar{\boldsymbol{x}})^T(\boldsymbol{x}_j - \bar{\boldsymbol{x}}) - 2 (\boldsymbol{x}_i - \bar{\boldsymbol{x}})^T(\boldsymbol{x}_j - \bar{\boldsymbol{x}}) 
\\
&= b_{ii} + b_{jj} - 2 b_{ij}
\end{aligned}
$$

所以由于$\hat{\boldsymbol{B}}=\boldsymbol{B}$，即可以直接得到$\hat{\boldsymbol{S}}=\boldsymbol{S}$。

## 3.

$$
\boldsymbol{\Sigma}_\varphi  \boldsymbol{v} = \lambda \boldsymbol{v}
\\
\Rightarrow \frac1n \sum_{i=1}^n \varphi(\boldsymbol{x}_i)\varphi(\boldsymbol{x}_i)^T\boldsymbol{v} = \lambda \boldsymbol{v}
$$

当$\lambda \ne 0$时：
$$
\boldsymbol{v} =\frac{1}{n\lambda} \sum_{i=1}^n \varphi(\boldsymbol{x}_i)[\varphi(\boldsymbol{x}_i)^T\boldsymbol{v}] =\frac{1}{n\lambda} \sum_{i=1}^n [\varphi(\boldsymbol{x}_i)^T\boldsymbol{v}]\varphi(\boldsymbol{x}_i)
$$
其中$[\varphi(\boldsymbol{x}_i)^T\boldsymbol{v}]$为标量，故可以表示为$\boldsymbol{v}= \sum_{i=1}^n\alpha_i \varphi(\boldsymbol{x}_i)$的形式。

