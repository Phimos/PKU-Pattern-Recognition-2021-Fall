# 聚类分析上机题

**姓名：** 甘云冲

**学号：** 2101213081

本次作业利用K-均值、核K-均值、RatioCut 和 Ncut 算法对给定的数据集进行聚类，并给出聚类结果图。

观察原始数据的类别分布如下：

<center>   
  <img src="Aggregation.png" width="300"/>
  <img src="Jain.png" width="300"/>
  <img src="Spiral.png" width="300"/>
</center>

对于不同的聚类算法，结果如下：

### K-均值算法

<center>   
  <img src="Aggregation_kmeans.png" width="300"/>
  <img src="Jain_kmeans.png" width="300"/>
  <img src="Spiral_kmeans.png" width="300"/>
</center>

### 核K-均值算法

<center>   
  <img src="Aggregation_kernel_kmeans.png" width="300"/>
  <img src="Jain_kernel_kmeans.png" width="300"/>
  <img src="Spiral_kernel_kmeans.png" width="300"/>
</center>

### RatioCut算法

<center>   
  <img src="Aggregation_ratio_cut.png" width="300"/>
  <img src="Jain_ratio_cut.png" width="300"/>
  <img src="Spiral_ratio_cut.png" width="300"/>
</center>

### NCut算法

<center>   
  <img src="Aggregation_ncut.png" width="300"/>
  <img src="Jain_ncut.png" width="300"/>
  <img src="Spiral_ncut.png" width="300"/>
</center>

可以发现，谱聚类算法（包括RatioCut和NCut算法）在通过参数调整之后能够得到接近真实label的聚类结果。

