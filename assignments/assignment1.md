# **Bayes** 决策与概率密度估计习题

**姓名：** 甘云冲

**学号：** 2101213081

## 1.

### (1)

$$
P(X|\text{Positive}) = \frac{P(\text{Positive}|X)P(X)}{P(\text{Positive}|X)P(X) + P(\text{Positive}|\neg X)P(\neg X)} = \frac{100\% *0.5\%}{100\% *0.5\% + 5\% * 99.5\%} = 9.1\%
$$

该人患有疾病X的概率为9.1%。

### (2)

$$
\begin{aligned}

P(X|\text{Positive, Positive})
&= \frac{P(\text{Positive, Positive}|X)P(X)}{P(\text{Positive, Positive}|X)P(X) + P(\text{Positive, Positive}|\neg X)P(\neg X)}
\\
&= \frac{P(\text{Positive}|X) ^ 2 P(X)}{P(\text{Positive}|X) ^ 2P(X) + P(\text{Positive}|\neg X) ^ 2P(\neg X)}
\\
&= \frac{(100\%)^2 *0.5\%}{(100\%)^2 *0.5\% + (5\%)^2 * 99.5\%} 
\\
&= 66.8\%
\end{aligned}
$$

该人患有疾病X的概率为66.8%。
$$
66.8\% * 10 > (1 - 66.8\%) * 1
$$
医生应当认为患者有病。



## 2.

### (1)

$$
\begin{aligned}
P(\omega_1|x) &= P(\omega_2|x)
\\
\Lrarr 
\frac{P(x|\omega_1)P(\omega_1)}{P(x|\omega_1)P(\omega_1) + P(x|\omega_2)P(\omega_2)} &= \frac{P(x|\omega_2)P(\omega_2)}{P(x|\omega_1)P(\omega_1) + P(x|\omega_2)P(\omega_2)}
\\
\Lrarr \frac{P(x|\omega_1)}{P(x|\omega_1) + P(x|\omega_2)} &= \frac{P(x|\omega_2)}{P(x|\omega_1) + P(x|\omega_2)}

\\
\Lrarr P(x|\omega_1) &= P(x|\omega_2)
\end{aligned}
$$

令$x=(a_1+a_2)/2$，可得到：
$$
P(x|\omega_1) = \frac{1}{\pi b}\frac{1}{1+(\frac{a_2 - a_1}{2b})^2} = P(x|\omega_2)
$$
得证结论。

### (2)

$$
\begin{aligned}
\lim_{x\rightarrow + \infty}P(\omega_1|x)
&=
\lim_{x\rightarrow + \infty}
\frac{P(x|\omega_1)P(\omega_1)}{P(x|\omega_1)P(\omega_1) + P(x|\omega_2)P(\omega_2)}
\\
&= \lim_{x\rightarrow + \infty}
\frac{(1 + (\frac{x - a_2}{b})^2)P(\omega_1)}{(1 + (\frac{x - a_2}{b})^2)P(\omega_1)+(1 + (\frac{x - a_1}{b})^2)P(\omega_2)}
\\
&= \frac{P(\omega_1)}{P(\omega_1) + P(\omega_2)}
\end{aligned}
$$

同理可得：
$$
\begin{aligned}
\lim_{x\rightarrow - \infty}P(\omega_1|x) = \frac{P(\omega_1)}{P(\omega_1) + P(\omega_2)}
\\
\lim_{x\rightarrow + \infty}P(\omega_2|x) = \frac{P(\omega_2)}{P(\omega_1) + P(\omega_2)}
\\
\lim_{x\rightarrow - \infty}P(\omega_2|x) = \frac{P(\omega_2)}{P(\omega_1) + P(\omega_2)}
\end{aligned}
$$

### (3)

不失一般性地假设$a_1 \le a_2$，由(1)中可以知道，最小误差判决边界为峰值的中点。
$$
\begin{aligned}
P(e) 
&= \int_{-\infty}^{(a_1+a_2)/2}p(x|\omega_2)p(\omega_2)dx + \int_{(a_1+a_2)/2}^{+\infty}p(x|\omega_1)p(\omega_1)dx
\\
&= \frac{1}{\pi b}b \int_{(a_2-a_1)/(2b)}^{\infty} \frac{1}{1+x^2}dx
\\
&=\frac{1}{\pi}\arctan(x)\bigg|_{\frac{a_2 - a_1}{2b}}^{\infty}
\\
&=\frac{1}{\pi} (\pi/2 - \arctan(\frac{a_2 - a_1}{2b}))
\\
&= \frac{1}{2} - \frac{1}{\pi}\arctan(\frac{a_2 - a_1}{2b})
\end{aligned}
$$
针对于$a_1>a_2$的情况同上可证明，所以最小概率误差为：
$$
P(e)=\frac{1}{2} - \frac{1}{\pi}\arctan\abs{\frac{a_2 - a_1}{2b}}
$$

### (4)

容易发现，当$a_1=a_2$时候，有$P(e)=1/2$为最大值，因为当$a_1=a_2$时候，两个类别并没有区别，只是在两类当中随机选择一类进行猜测，所以错误率为1/2。



## 3.

### (1)

$$
\hat{\mu} = \frac{5.0+7.0+9.0+11.0+13.0}{5} = 9.0
$$

$$
\hat{\sigma}^2 = \frac{(5.0-9.0)^2+(7.0-9.0)^2+(9.0-9.0)^2+(11.0-9.0)^2+(13.0-9.0)^2}{5} = 8.0
$$

### (2)

b越小似然函数越大，但是b不能小于8，这里参数b应当估计为8

### (3)

由于先验概率相等只需要对比两类下样本的pdf。
$$
p(x|\omega_1) = \frac{1}{\sqrt{2\pi*8}} e^{-\frac{(6-9)^2}{2 * 8}} \approx 0.08
$$

$$
p(x|\omega_2) = 1/8 = 0.125
$$

应当判定为第二类。

## 4.

### (1)

$$
\begin{aligned}
\sum_{i=1}^{n}\ln p(x_i|\theta, \sigma) 
&=
\sum_{i=1}^{n}-\frac{(\ln x_i - \theta)^2}{2\sigma^2}  - \ln \sigma x_i\sqrt{2\pi}
\end{aligned}
$$

$\theta$的最大似然估计：
$$
\begin{aligned}
\frac{d \sum_{i=1}^{n}\ln p(x_i|\theta, \sigma) }{d\theta} &= 0
\\
\sum_{i=1}^n (\ln x_i - \theta) &= 0
\\
\hat{\theta}_{ML} = \frac{\sum_{i=1}^n \ln x_i}{n}
\end{aligned}
$$

### (2)

$\sigma$的最大似然估计：
$$
\begin{aligned}
\frac{d \sum_{i=1}^{n}\ln p(x_i|\theta, \sigma) }{d\sigma} &= 0
\\
\sum_{i=1}^n \frac{(\ln x_i - \theta)^2}{\sigma^3} - \frac{x_i \sqrt{2\pi}}{\sigma} &= 0
\\
\hat\sigma^2 = \frac{\sum_{i=1}^n (\ln x_i - \hat{\theta})^2}{\sqrt{2\pi}\sum_{i=1}^n x_i}
\end{aligned}
$$

### (3)

$\theta$的MAP估计：
$$
\sum_{i=1}^n \ln p(x|\theta) + \ln p(\theta)
\\
= \sum_{i=1}^{n} \left(-\frac{(\ln x_i - \theta)^2}{2\sigma^2}  - \ln \sigma x_i\sqrt{2\pi} \right) - \frac{1}{2}\ln(2\pi \sigma_0^2) - \frac{1}{2} (\frac{\theta - \theta_0}{\sigma_0})^2
$$
于是：
$$
\begin{aligned}
\frac{d\left(\sum_{i=1}^n \ln p(x|\theta) + \ln p(\theta)\right) }{d\theta} &= 0
\\
\sum_{i=1}^n \frac{\ln x_i - \theta}{\sigma^2} -  \frac{\theta - \theta_0}{\sigma_0^2} &= 0
\\
\sigma_0^2\sum_{i=1}^n \ln x_i - n\sigma_0^2\theta - \sigma^2 \theta + \sigma^2 \theta_0 &= 0
\\
\hat \theta _{MAP} &= \frac{\sigma_0^2\sum_{i=1}^n \ln x_i  + \sigma^2 \theta_0}{n\sigma_0^2 + \sigma^2}
\end{aligned}
$$

