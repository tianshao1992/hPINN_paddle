import numpy as np
import paddle
import paddle.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def numpy_32(x):
    if isinstance(x, (list, tuple)):
        y = []
        for xx in x:
            y.append(xx.detach().cpu().numpy())
    else:
        y = x.detach().cpu().numpy()
    return y

def tensor_32(x):
    if isinstance(x, (list, tuple)):
        y = []
        for xx in x:
            y.append(paddle.to_tensor(xx, dtype='float32', place='gpu:0'))
    else:
        y = paddle.to_tensor(x, dtype='float32', place='gpu:0')
    return y

###################自动微分求梯度以及Jacobian矩阵######################
def gradients(y, x, order=1, create=True):
    if order == 1:
        return paddle.grad(y, x, create_graph=create, retain_graph=True)[0]
    else:
        return paddle.stack([paddle.grad([y[:, i].sum()], [x], create_graph=True, retain_graph=True)[0]
                            for i in range(y.shape[1])], axis=-1)

###################多个单一输出的神经网络########################
class PaddleModel_multi(nn.Layer):
    def __init__(self, planes,  active=nn.Tanh()):
        super(PaddleModel_multi, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.LayerList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1],
                                       weight_attr=nn.initializer.XavierNormal()))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
            # self.layers[-1].apply(initialize_weights)

    def forward(self,in_var):
        # in_var = self.x_norm.norm(in_var)     正则化
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return paddle.concat(y, axis=-1)

    def loadmodel(self, File):
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取  ! 可能有问题
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
            Log_loss = checkpoint['log_loss'].tolist()
            return Log_loss
        except:
            print("load model failed！ start a new model.")
            return []


if __name__ == '__main__':
    Net_model = PaddleModel_multi(planes=[2] + [64] * 4 + [4])
    x = paddle.ones([1000, 2])
    y = Net_model(x)
    print(y)
