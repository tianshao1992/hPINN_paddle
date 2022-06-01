# Paddle 论文复现挑战赛（科学计算方向131）
### 
- ## **论文解读**

  原文：[PHYSICS-INFORMED NEURAL NETWORKS WITH HARD CONSTRAINTS FOR INVERSE DESIGN](https://arxiv.org/pdf/2102.04626.pdf)

  参考：[hpinn-DeepXDE](https://github.com/lululxvi/hpinn)

  - 本文采用了physics-informed neural networks(PINNs)解决工业领域的难点之一——反设计问题；本质上是变密度法拓扑优化的PINNs实现，除了PDE本身待求解的物理场以外，还需求解拓扑优化的密度场，该密度场描述材料是否需要特定位置填充，通过密度场分布给出最终的设计方案，一般用于增材制造、3D打印的结构设计中，是计算固体力学中的重要研究方向之一。

  - 在PINNs中实现时，利用神经网络同时预测物理场以及密度场，因此需要根据具体设计问题引入待优化的目标函数使方程封闭；该问题也是不适定的偏微分方程中的一类，可以凸显了PINN方法解决不适定问题中的灵活性优势。相对于传统的PINNs，作者处理的技巧主要包括了两点（hPINNs——physics-informed neural networks with hard constraints）：

    - 在网络结构设计，hPINNs采用了多个全连接层，**每个全连接层预测一个物理场输出**。
    - PDE的边界条件不作为软约束引入优化损失中，而是通过设计专用的**硬约束函数**直接保证在边界条件的成立。

    - 引入PDE损失逐渐增大的惩罚项以及**Lagrange增强方法**优化损失函数，相对于传统的优化方法可以达到更好的效果。

  - 作者选择了光学全息以及Stokes流体中的反设计问题验证了方法的有效性。要求复现的算例是光学全息中的反设计问题。

- ## 代码说明
- [hpinn-paddle AI studio](https://aistudio.baidu.com/aistudio/projectdetail/4117361?contributionType=1&shared=1)

  - train_hpinn_horo 中为实现的lagrange增强方法求解带有硬约束的PDE及其反设计问题训练过程 （**训练模型运行**）
  - valid_hpinn_horo中为结果验证以及原文中Fig. 7中图片绘制 （**验证结果及出图**）
  - **fig文件夹**中为原始论文结果相关图片，复现模型文件以及结果文件见根目录的**res文件夹**
    - \train  训练集数据（input_train.txt） & 训练过程的可视化
    - \model 所有训练过程中的神经网络模型保存
    - \data 所有训练过程中的数据保存
    - \figure 验证数据与复现论文对比的图片
  - basic_model.py 中为实现的多个全连接网络预测多种输出

  - horo_model.py 中为本问题所涉及到的参数以及PDE残差损失、目标函数计算

  - process_data.py 中为求解域离散点采样方法

  - visual_data.py 为训练过程的数据可视化

- ## 环境依赖

  > numpy == 1.22.3 \
  > scipy == 1.8.0  \
  > scikit-optimize == 0.9.0 \
  > paddlepaddle-gpu == 2.3.0 \
  > paddle==1.0.2 \
  > matplotlib==3.5.1 \
  > seaborn==0.11.2 

- ## 复现结果

|      |  PDE残差损失   | 目标函数 |
| :--: | :------------: | :------: |
| 论文 |   $10^{-4}$    | $0.055$  |
| 复现 | $3.97×10^{-5}$ | $0.055$  |

| 论文Fig.6                                                    | 复现Fig.6                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20220526132137179](/fig/fig6A.jpg) | ![valid_Loss_Fig6_A](/res/hpinn_horo_mu_2_lag/figure/valid_Loss_Fig6_A.jpg) |
| ![image-20220526132205167](/fig/fig6B.jpg) | ![valid_Objetive_Fig6_B](/res/hpinn_horo_mu_2_lag/figure/valid_Objetive_Fig6_B.jpg) |
| ![image-20220526132221667](/fig/fig6C.jpg) | ![valid_Fig 6C](/res/hpinn_horo_mu_2_lag/figure/valid_Fig_6C.png) |
| ![image-20220526132236712](/fig/fig6D.jpg) | ![valid_Fig 6C](/res/hpinn_horo_mu_2_lag/figure/valid_lambda_Fig6_D.jpg) |
| ![image-20220526132249136](/fig/fig6E.jpg) | ![valid_lambda_Fig6_E](/res/hpinn_horo_mu_2_lag/figure/valid_lambda_Fig6_E.jpg) |
| ![image-20220526132257938](/fig/fig6F.jpg) | ![valid_lambda_Fig6_F](/res/hpinn_horo_mu_2_lag/figure/valid_lambda_Fig6_F.jpg) |



- ## 存在问题
  - 科学计算中需要计算高阶微分，不仅是tanh、sigmoid等简单函数，paddle 目前无法支持sin cos exp 等算子的高阶微分计算；导致本算例中复现相对麻烦。
  - PINN中，在Adam优化结束后切换L-BGFS可以大幅降低，提升PDE求解精度，目前paddle的L-BFGS无法使用，本工作中均采用Adam，虽然可以实现要求，但根据经验，对其他问题未必奏效。

- ## paddle 目前无法支持算子的高阶微分计算解决方案 

![image-20220526200528982](/fig/forward.png)

- 考虑featuretransform里具有sin() cos() 以及 HardConstriant里有exp() 等Paddle暂时无法支持高阶微分的算子，因此计算$\nabla_x f(\pmb{x})$，$\nabla_x^T \nabla_x f(\pmb{x})$，（其中$f$指代电场实部或虚部的标量场，$\pmb{x}$指代空间坐标的向量，其余为中间涉及的映射），需要将无法自动微分的映射剥离开，利用手动微分计算这些部分的高阶导数，最后再用于PDE中的残差损失，上述计算的前向过程如下式所示：

$$
\begin{aligned}
\pmb{h}(\pmb{x}) &= featuretransform(\pmb{x})\\
w(\pmb{h}) &= NeuralNetwork(\pmb{h})\\
t(\pmb{x}) &= HardConstraint(\pmb{x})\\
f(t, \pmb{w}) &= Outputtransform(t, w) =t(\pmb{x})\cdot w(\pmb{x})\\
\end{aligned}
$$

   其中，$w(\pmb{h})$可中含有tanh()激活函数，paddle已实现其高阶微分计算方式，利用复合函数、乘积的微分运算以及链式法则，得到：

- 一阶微分$\nabla_x f(\pmb{x})$计算		如下：

$$
\begin{aligned}
\nabla_x f(\pmb{x})&=t(\pmb{x})\cdot\nabla_x w(\pmb{x}) + w(\pmb{x})\cdot\nabla_x t(\pmb{x})\\
\nabla_x w(\pmb{x})&=\nabla_h w(\pmb{h}) \nabla_x \pmb{h}(\pmb{x})\\
\end{aligned}
$$
- 二阶微分$\nabla_x^T \nabla_x f(\pmb{x})$计算如下：

$$
\begin{aligned}
\nabla_x^T \nabla_x f(\pmb{x})&=\nabla_x^T t(\pmb{x})\cdot\nabla_x w(\pmb{x}) + t(\pmb{x})\cdot\nabla_x^T \nabla_x w(\pmb{x}) +  \nabla_x^T w(\pmb{x})\nabla_x t(\pmb{x}) + w(\pmb{x})\cdot\nabla_x^T \nabla_x t(\pmb{x}) \\
\nabla_x^T \nabla_x w(\pmb{x})&=\nabla_x^T \pmb{h}(\pmb{x}) \nabla_h^T \nabla_h w(\pmb{h}) \nabla_x \pmb{h}(\pmb{x}) + \nabla_h w(\pmb{h}) \nabla_h^T\nabla_x h(\pmb{x})
\end{aligned}
$$
- 考虑可以采用paddle提供的自动微分计算$\nabla_h \pmb{w}(\pmb{h})$以及二阶导$\nabla_h^T \nabla_h w(\pmb{h})$，其余部分计算手动微分即可获得，$\nabla_x h(\pmb{x})$，$\nabla_h^T\nabla_x h(\pmb{x})$ 以及 $\nabla_x t(\pmb{x})$，$\nabla_h^T\nabla_x t(\pmb{x})$，具体代码见feature_backward 、output_barkward中，对于上述微分中涉及矩阵的运算采用Broadcast机制即可实现。
- 显然，这样的方式对二阶导$\nabla_h^T \nabla_h w(\pmb{h})$的计算大幅增加了自动微分反向传播graph的存储量和计算量，该矩阵的维度为batch_size×15×15（$\pmb{h}$的维度为15），w需要计算实部虚部因此还要再扩大2倍
- 上述计算过程的详细代码如下：


``` python
DtDx, D2tDx = self.output_backward(x)
DhDx, D2hDx = self.feature_backward(x)
DwRDh = gradients(w[..., :1].sum(), h)
DwIDh = gradients(w[..., 1:2].sum(), h)
D2wRDh = gradients(DwRDh, h, order=2)
D2wIDh = gradients(DwIDh, h, order=2)

DwRDx = (DwRDh.unsqueeze(-2) @ DhDx).squeeze()
DwIDx = (DwIDh.unsqueeze(-2) @ DhDx).squeeze()

dReEdx = t * DwRDx + w[..., :1] * DtDx
dImEdx = t * DwIDx + w[..., 1:2] * DtDx

D2wRDx = DhDx.transpose([0, 2, 1]) @ D2wRDh @ DhDx + (DwRDh.unsqueeze(-2) @ D2hDx).reshape((-1, 2, 2))
D2wIDx = DhDx.transpose([0, 2, 1]) @ D2wIDh @ DhDx + (DwIDh.unsqueeze(-2) @ D2hDx).reshape((-1, 2, 2))

D2ReDx = DtDx.unsqueeze(-1) @ DwRDx.unsqueeze(-2) + t.unsqueeze(-1) * D2wRDx + \
DwRDx.unsqueeze(-1) @ DtDx.unsqueeze(-2) + (w[:, :1] * D2tDx).reshape((-1, 2, 2))
D2ImDx = DtDx.unsqueeze(-1) @ DwIDx.unsqueeze(-2) + t.unsqueeze(-1) * D2wIDx + \
DwIDx.unsqueeze(-1) @ DtDx.unsqueeze(-2) + (w[:, 1:2] * D2tDx).reshape((-1, 2, 2))
```
