import numpy as np
import paddle
import paddle.nn as nn
from basic_model import PaddleModel_multi, gradients, numpy_32, tensor_32


###################问题定义相关参数#############:1]########
BOX = np.array([[-2, -2], [2, 3]])
DPML = 1
OMEGA = 2 * np.pi
SIGMA0 = -np.log(1e-20) / (4 * DPML ** 3 / 3)
l_BOX = BOX + np.array([[-DPML, -DPML], [DPML, DPML]])


###################问题定义相关函数########################
def PML(X):
    def sigma(x, a, b):
        """sigma(x) = 0 if a < x < b, else grows cubically from zero.
        """

        def _sigma(d):
            return SIGMA0 * d ** 2 * np.heaviside(d, 0)

        return _sigma(a - x) + _sigma(x - b)

    def dsigma(x, a, b):
        def _sigma(d):
            return 2 * SIGMA0 * d * np.heaviside(d, 0)

        return -_sigma(a - x) + _sigma(x - b)

    sigma_x = sigma(X[:, :1], BOX[0][0], BOX[1][0])
    AB1 = 1 / (1 + 1j / OMEGA * sigma_x) ** 2
    A1, B1 = AB1.real, AB1.imag

    dsigma_x = dsigma(X[:, :1], BOX[0][0], BOX[1][0])
    AB2 = -1j / OMEGA * dsigma_x * AB1 / (1 + 1j / OMEGA * sigma_x)
    A2, B2 = AB2.real, AB2.imag

    sigma_y = sigma(X[:, 1:], BOX[0][1], BOX[1][1])
    AB3 = 1 / (1 + 1j / OMEGA * sigma_y) ** 2
    A3, B3 = AB3.real, AB3.imag

    dsigma_y = dsigma(X[:, 1:], BOX[0][1], BOX[1][1])
    AB4 = -1j / OMEGA * dsigma_y * AB3 / (1 + 1j / OMEGA * sigma_y)
    A4, B4 = AB4.real, AB4.imag
    return A1, B1, A2, B2, A3, B3, A4, B4


def J(x):
    # Approximate the delta function
    y = x[:, 1:] + 1.5
    # hat function of width 2 * h
    # h = 0.5
    # return 1 / h * np.maximum(1 - np.abs(y / h), 0)
    # normal distribution of width ~2 * 2.5h
    h = 0.2
    return 1 / (h * np.pi ** 0.5) * np.exp(-((y / h) ** 2)) * (np.abs(y) < 0.5)
    # constant function of width 2 * h
    # h = 0.25
    # return 1 / (2 * h) * (np.abs(y) < h)


################holography问题分析的神经网络，继承自basic model########################
class Net(PaddleModel_multi):
    def __init__(self, planes, X):
        super(Net, self).__init__(planes, active=nn.Tanh())

        self.X = X.detach().cpu().numpy()
        self.condition = tensor_32(np.logical_and(self.X[..., 1:] < 0, self.X[..., 1:] > -1).astype(np.float32))
        [self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.A4, self.B4] = tensor_32(PML(self.X))
        self.J = tensor_32(J(self.X))
        self.f1 = tensor_32(np.heaviside((self.X[..., :1] + 0.5) * (0.5 - self.X[..., :1]), 0.5))
        self.f2 = tensor_32(np.heaviside((self.X[..., 1:] - 1) * (2 - self.X[..., 1:]), 0.5))


    ########################输入坐标、输出转换为频域形式########################
    def feature_transform(self, inputs):
        # Periodic BC in x
        P = BOX[1][0] - BOX[0][0] + 2 * DPML # 周期长度
        w = 2 * np.pi / P
        x, y = inputs[..., :1], inputs[..., 1:]
        return paddle.concat(
            (
                paddle.cos(w * x),
                paddle.sin(w * x),
                paddle.cos(2 * w * x),
                paddle.sin(2 * w * x),
                paddle.cos(3 * w * x),
                paddle.sin(3 * w * x),
                paddle.cos(4 * w * x),
                paddle.sin(4 * w * x),
                paddle.cos(5 * w * x),
                paddle.sin(5 * w * x),
                paddle.cos(6 * w * x),
                paddle.sin(6 * w * x),
                y,
                paddle.cos(OMEGA * y),
                paddle.sin(OMEGA * y),
            ),
            axis=-1,
        )

    ########################手动计算feature transform的一阶以及二阶微分 ########################
    def feature_backward(self, inputs):
        # Periodic BC in x
        P = BOX[1][0] - BOX[0][0] + 2 * DPML # 周期长度
        w = 2 * np.pi / P
        x, y = inputs[..., :1], inputs[..., 1:]
        D1 = paddle.stack(
            (
                paddle.concat((-w*paddle.sin(w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( w*paddle.cos(w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-2*w*paddle.sin(2 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( 2*w*paddle.cos(2 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-3*w*paddle.sin(3 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( 3*w*paddle.cos(3 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-4*w*paddle.sin(4 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( 4*w*paddle.cos(4 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-5*w*paddle.sin(5 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( 5*w*paddle.cos(5 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-6*w*paddle.sin(6 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( 6*w*paddle.cos(6 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( paddle.zeros_like(x), paddle.ones_like(x)), axis=-1),
                paddle.concat(( paddle.zeros_like(x), -OMEGA * paddle.sin(OMEGA * y)), axis=-1),
                paddle.concat(( paddle.zeros_like(x),  OMEGA * paddle.cos(OMEGA * y)), axis=-1),
            ),
            axis=1,
        )

        D2 = paddle.ones((inputs.shape[0], 15, 4), dtype=paddle.float32)
        M = paddle.stack(
            (
                paddle.concat((-w*w*paddle.cos(w *x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*paddle.sin(w *x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*2*2*paddle.cos(2 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*2*2*paddle.sin(2 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*3*3*paddle.cos(3 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*3*3*paddle.sin(3 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*4*4*paddle.cos(4 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*4*4*paddle.sin(4 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*5*5*paddle.cos(5 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*5*5*paddle.sin(5 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*6*6*paddle.cos(6 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat((-w*w*6*6*paddle.sin(6 * w * x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( paddle.zeros_like(x), paddle.zeros_like(x)), axis=-1),
                paddle.concat(( paddle.zeros_like(x), -OMEGA * OMEGA * paddle.cos(OMEGA * y)), axis=-1),
                paddle.concat(( paddle.zeros_like(x), -OMEGA * OMEGA * paddle.sin(OMEGA * y)), axis=-1),
            ),
            axis=1,
        )
        D2[..., 0] = M[..., 0]
        D2[..., 3] = M[..., 1]

        return D1, D2

    ########################输入坐标以及神经网络输出，施加硬约束并输出电场E以及epsilon ########################
    def output_transform(self, inn_var, out_var):
        x, y = inn_var[..., :1], inn_var[..., 1:]

        # 1 <= eps <= 12
        eps = nn.functional.sigmoid(out_var[..., -1:])*11 + 1

        # Zero Dirichlet BC
        a, b = BOX[0][1] - DPML, BOX[1][1] + DPML
        t = (1 - paddle.exp(a - y)) * (1 - paddle.exp(y - b))
        E = t * out_var[..., :2]
        # condition = np.logical_and(X[:, 1:] < 0, X[:, 1:] > -1).astype(np.float32)
        # eps = out_var[..., -1:] * tensor_32(condition) + 1 - tensor_32(condition)

        return paddle.concat([E, eps], axis=-1), t

    ########################手动计算output transform的一阶以及二阶微分 ########################
    def output_backward(self, inn_var):
        x, y = inn_var[..., :1], inn_var[..., 1:]
        # Zero Dirichlet BC
        a, b = BOX[0][1] - DPML, BOX[1][1] + DPML

        dx = paddle.zeros_like(x)
        dy = paddle.exp(a - y) * (1 - paddle.exp(y - b)) + (1 - paddle.exp(a - y)) * (- paddle.exp(y - b))

        dxx = paddle.zeros_like(x)
        dxy = paddle.zeros_like(x)
        dyx = paddle.zeros_like(x)
        dyy = -paddle.exp(a - y) * (1 - paddle.exp(y - b)) + paddle.exp(a - y) * (- paddle.exp(y - b)) + \
               paddle.exp(a - y) * (- paddle.exp(y - b)) + (1 - paddle.exp(a - y)) * (- paddle.exp(y - b))

        return paddle.concat((dx, dy), axis=-1), paddle.concat((dxx, dxy, dyx, dyy), axis=-1)


    ########################计算PDE残差， X 为坐标，(paddle支持sin cos exp等算子高阶微分时调用) ############################
    def pde(self, inn_var, out_var):

        # ReE, ImE = out_var[..., (0,)], out_var[..., (1,)]
        ReE, ImE = out_var[..., :1], out_var[..., 1:2]
        # 取出在区域Omega2 处的eps, 并使得其他位置的eps = 1
        eps = out_var[..., -1:] * self.condition + 1 - self.condition

        dReEda = gradients(ReE.sum(), inn_var)
        dReE_x, dReE_y = dReEda[..., :1], dReEda[..., 1:]
        dReE_xx = gradients(dReE_x.sum(), inn_var)[..., :1]
        dReE_yy = gradients(dReE_y.sum(), inn_var)[..., 1:]

        dImEda = gradients(ImE.sum(), inn_var)
        dImE_x, dImE_y = dImEda[..., :1], dImEda[..., 1:]
        dImE_xx = gradients(dImE_x.sum(), inn_var)[..., :1]
        dImE_yy = gradients(dImE_y.sum(), inn_var)[..., 1:]

        loss_Re = (
                (self.A1 * dReE_xx + self.A2 * dReE_x + self.A3 * dReE_yy + self.A4 * dReE_y) / OMEGA
                - (self.B1 * dImE_xx + self.B2 * dImE_x + self.B3 * dImE_yy + self.B4 * dImE_y) / OMEGA
                + eps * OMEGA * ReE
        )
        loss_Im = (
                (self.A1 * dImE_xx + self.A2 * dImE_x + self.A3 * dImE_yy + self.A4 * dImE_y) / OMEGA
                + (self.B1 * dReE_xx + self.B2 * dReE_x + self.B3 * dReE_yy + self.B4 * dReE_y) / OMEGA
                + eps * OMEGA * ImE
                + self.J
                )
        # return loss_Re, loss_Im
        return paddle.concat([loss_Re, loss_Im], axis=-1) # augmented_Lagrangian



    ########################计算PDE残差， X为输入坐标(paddle无法支持sin cos exp等算子高阶微分时调用) ############################
    def pde_(self, x, h, w, t, f):
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

        # AB = tensor_32(PML(X))
        ReE, ImE = f[..., :1], f[..., 1:2]

        # 取出在区域Omega2 处的eps, 并使得其他位置的eps = 1
        eps = f[..., 2:] * self.condition + 1 - self.condition

        # dReEda = gradients(ReE, x)
        dReE_x, dReE_y = dReEdx[..., :1], dReEdx[..., 1:]
        dReE_xx, dReE_yy = D2ReDx[..., 0, :1], D2ReDx[..., 1, 1:]

        # dImEda = gradients(ImE, x)
        dImE_x, dImE_y = dImEdx[..., :1], dImEdx[..., 1:]
        dImE_xx, dImE_yy = D2ImDx[..., 0, :1], D2ImDx[..., 1, 1:]

        loss_Re = (
                (self.A1 * dReE_xx + self.A2 * dReE_x + self.A3 * dReE_yy + self.A4 * dReE_y) / OMEGA
                - (self.B1 * dImE_xx + self.B2 * dImE_x + self.B3 * dImE_yy + self.B4 * dImE_y) / OMEGA
                + eps * OMEGA * ReE
        )
        loss_Im = (
                (self.A1 * dImE_xx + self.A2 * dImE_x + self.A3 * dImE_yy + self.A4 * dImE_y) / OMEGA
                + (self.B1 * dReE_xx + self.B2 * dReE_x + self.B3 * dReE_yy + self.B4 * dReE_y) / OMEGA
                + eps * OMEGA * ImE
                + self.J
        )

        # return loss_Re, loss_Im
        return paddle.concat([loss_Re, loss_Im], axis=-1) # augmented_Lagrangian

    ######################## 优化目标函数，J:############################
    ## 需要提取出在g3区域的位置
    def optim_func(self, out_var, if_reduce=False):
        bound = out_var.shape[0]
        j = out_var[:, :1] ** 2 + out_var[:, 1:2] ** 2 - self.f1[:bound]*self.f2[:bound]
        if if_reduce:
            return paddle.mean(j)
        else:
            return j
