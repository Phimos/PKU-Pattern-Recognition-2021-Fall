# 正则化与重构核Hilbert空间习题

**姓名：** 甘云冲

**学号：** 2101213081

## 1.

### (1)

$$
\begin{cases}
3c_2 +4c_3 = 1
\\
3c_1 + c_3 = 4
\\
4c_1 + c_2 = 3
\end{cases}
\Rightarrow

\begin{cases}
c_1 = 1
\\
c_2 = -1
\\
c_3 = 1
\end{cases}
$$

### (2)

定义：
$$
\mathbf{\hat{\mathbf{x}}} = [x_1, x_2, x_3]^T
$$

重写核函数为如下形式：

$$
K(x, z) = |x - z|
$$

利用核矩阵表示原函数为：


$$
f(x) = \mathbf{K}(x, \hat{\mathbf{x}})_{1\times3}\mathbf{c}_{3\times1}
$$

那么同理可表示$b_i$如下：
$$
b_i(x) = \mathbf{K}(x, \hat{\mathbf{x}})_{1\times3}\mathbf{h_i}_{3\times1}
$$

表示$f(x)$如下：

$$
f(x) = \mathbf{K}(x, \hat{\mathbf{x}})_{1\times3} \mathbf{H}_{3\times3}\mathbf{y}_{3\times1}
$$
其中$H_{ij} = h_j^i$，又由于$y_i = f(x_i)$

$$
f(x) = \mathbf{K}(x, \hat{\mathbf{x}})_{1\times3} \mathbf{H}_{3\times3}\mathbf{K}(\hat{\mathbf{x}}, \hat{\mathbf{x}})_{3\times3}\mathbf{c}_{3\times1}
$$

与原式做对比，可以发现只要$\mathbf{H}$满足如下条件，即可以保证与原式意义相同，于是得到最终的结果：
$$
\mathbf{H} = \mathbf{K}(\hat{\mathbf{x}}, \hat{\mathbf{x}})_{3\times3}^{-1}
$$
计算核矩阵的数值表示如下所示：
$$
\mathbf{K}(\hat{\mathbf{x}}, \hat{\mathbf{x}}) =  \left[
 \begin{matrix}
   0 & 3 & 4 \\
   3 & 0 & 1 \\
   4 & 1 & 0
  \end{matrix}
  \right]
$$
所以可以得到对应的$\mathbf{H}$矩阵：
$$
\mathbf{H} = \left[
 \begin{matrix}
   -\frac{1}{24} & \frac{1}{6} & \frac{1}{8} \\
   \frac16 & -\frac23 & \frac12 \\
   \frac18 & \frac12 & -\frac38
  \end{matrix}
  \right]
$$

将其带入原式便可以得到$b_i(x)$，两种核函数的关联如前文所示。

## 2.

由核函数定义可以知道：
$$
K(\mathbf{x}, \mathbf{z})=\langle\phi(\mathbf{x}), \phi(\mathbf{z})\rangle
$$

$$
\begin{aligned}
&K(\mathbf{x}, \mathbf{z}) + K(0, \mathbf{z})+ K(\mathbf{x}, 0)+ K(0, 0)
\\
=&\langle\phi(\mathbf{x}), \phi(\mathbf{z})\rangle + \langle\phi(0), \phi(\mathbf{z})\rangle+\langle\phi(\mathbf{x}), \phi(0)\rangle+\langle\phi(0), \phi(0)\rangle
\\
=&\langle\phi(\mathbf{x}) + \phi(0), \phi(\mathbf{z})+\phi(0)\rangle
\\
=&\langle\hat\phi(\mathbf{x}), \hat\phi(\mathbf{z})\rangle
\end{aligned}
$$

故也为核函数。

## 3.

$$
\begin{aligned}
\varphi(x)^{T} \varphi(z) &= \frac12 + \sum_{i=1}^k \left[\cos(ix)\cos(iz) +\sin(ix)\sin(iz) \right]
\\
&= \frac12 + \sum_{i=1}^k \left[\frac12(\cos(ix+iz)) + \cos(ix-iz)) -\frac12(\cos(ix+iz)) - \cos(ix-iz))\right]
\\
&= \frac12 + \sum_{i=1}^k \cos(i(x-z))
\\
&= \frac{1}{\sin(\frac{x-z}{2})} \left[\frac{1}{2}\sin(\frac{x-z}{2})  +  \sum_{i=1}^k\sin(\frac{x-z}{2})\cos(i(x-z))\right]
\\
&= \frac{1}{\sin(\frac{x-z}{2})} \left[\frac{1}{2}\sin(\frac{x-z}{2})  +  \frac12\sum_{i=1}^k\left(\sin\left(\left(i+\frac12\right)(x-z)\right) - \sin\left(\left(i-\frac12\right)(x-z)\right)\right)\right]
\\
&= \frac{\sin\left(\left(k+\frac12\right)(x-z)\right)}{2\sin(\frac{x-z}{2})}
\end{aligned}
$$

## 4.

原优化目标改写为矩阵形式：
$$
L= \|\mathbf{X}\mathbf{w} - \tilde{\mathbf{y}} \|^2+\lambda\|\mathbf{w}\|^{2}
$$
求偏导：
$$
\begin{aligned}
\frac{\part L}{\part \mathbf{w}} &= 2\mathbf{X}^T(\mathbf{X}\mathbf{w} - \tilde{\mathbf{y}}) + 2\lambda \mathbf{w}
\\
&= 2(\mathbf{X}^T\mathbf{X} + \lambda)\mathbf{w} - 2\mathbf{X}^T\tilde{\mathbf{y}} = 0
\end{aligned}
$$

$$
\begin{aligned}
\Rightarrow \qquad(\mathbf{X}^T\mathbf{X} + \lambda)\mathbf{w} &= \mathbf{X}^T\tilde{\mathbf{y}}
\\
(\mathbf{X}^T\mathbf{X} + \lambda)\mathbf{X}^T\mathbf{\alpha} &= \mathbf{X}^T\tilde{\mathbf{y}}
\\
\mathbf{X}^T(\mathbf{X}\mathbf{X}^T + \lambda \mathbf{I})\alpha &= \mathbf{X}^T\tilde{\mathbf{y}}
\\
\mathbf{X}^T(\mathbf{G}+ \lambda \mathbf{I})\alpha &= \mathbf{X}^T\tilde{\mathbf{y}}
\end{aligned}
$$

所以解为：
$$
\mathbf{\alpha} = (\mathbf{G}+ \lambda \mathbf{I})^{-1}\tilde{\mathbf{y}}
$$
对于核函数表达形式，设核矩阵为$\mathbf{K}$，有$\mathbf{K}_{ij} = \langle\mathbf{x}_i, \mathbf{x}_j\rangle$，核函数表达形式为：
$$
\mathbf{\alpha} =(\mathbf{K}+ \lambda \mathbf{I})^{-1}\tilde{\mathbf{y}}
$$
