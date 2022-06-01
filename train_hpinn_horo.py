import numpy as np
import paddle
import paddle.nn as nn

import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio

from process_data import gen_dataset
from basic_model import numpy_32, tensor_32
from horo_model import Net
import visual_data


# os.environ['NUMEXPR_MAX_THREADS'] = '20'


def train_batch(x, bound, model, loss, weight, optimizer, log_loss, lambda_Re, lambda_Im,):
    optimizer.clear_grad()

    h = model.feature_transform(x)
    w = model(h)
    f, t = model.output_transform(x, w)
    res = model.pde_(x, h, w, t, f)
    opt = model.optim_func(f[:bound]) # opt_loss 仅在 g3 区域计算

    eqs_loss1 = loss(res[bound:, :1], paddle.zeros_like(res[bound:, :1], dtype='float32'))
    eqs_loss2 = loss(res[bound:, 1:], paddle.zeros_like(res[bound:, 1:], dtype='float32'))
    lag_loss1 = paddle.mean(res[bound:, :1] * lambda_Re)
    lag_loss2 = paddle.mean(res[bound:, 1:] * lambda_Im)
    opt_loss = loss(opt, paddle.zeros_like(opt, dtype='float32'))

    loss_batch = weight[0] * eqs_loss1 + weight[1] * eqs_loss2 + weight[2] * lag_loss1 + weight[3] * lag_loss2 + weight[4] * opt_loss
    loss_batch.backward()

    log_loss.append([eqs_loss1.item(), eqs_loss2.item(), lag_loss1.item(), lag_loss2.item(), opt_loss.item(), loss_batch.item()])
    optimizer.step()


def inference(inn_var, model, if_res=False):

    h = model.feature_transform(inn_var)
    w = model(h)
    f, t = model.output_transform(inn_var, w)
    if if_res:
        res = model.pde_(inn_var, h, w, t, f)
    else:
        res = paddle.zeros_like(inn_var)
    return numpy_32(f), numpy_32(res)


def train(name, epoch, input_train, bound, model, loss, loss_weight, optimizer, log_loss,
          lambda_Re, lambda_Im, display_epoch=1000):

    star_time = time.time()
    for iter in range(epoch):
        inn_var = input_train
        inn_var.stop_gradient = False

        lr_net = optimizer.get_lr()

        train_batch(inn_var, bound, model, loss, loss_weight, optimizer, log_loss, tensor_32(lambda_Re), tensor_32(lambda_Im),)

        if iter > 0 and iter % display_epoch == 0:
            print(' iter: {:6d}, lr_net: {:.3e},  cost: {:.2f} , total loss: {:.3e}, '
                  'eqs_loss_1: {:.3e}, eqs_loss_2: {:.3e},  lag_loss_1: {:.3e}, lag_loss_2: {:.3e},  opt_loss: {:.3e} '.
                  format(iter, lr_net,  time.time() - star_time, log_loss[-1][-1],
                         log_loss[-1][0], log_loss[-1][1], log_loss[-1][2], log_loss[-1][3], log_loss[-1][4]))


            plt.figure(1, figsize=(15, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss_1')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'eqs_loss_2')
            # Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'lag_loss_1')
            # Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'lag_loss_2')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'opt_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'total_loss')
            plt.savefig(os.path.join(Visual.log_dir, 'log_loss.svg'))

            star_time = time.time()

        if iter > 0 and iter % display_epoch == 0:

            output_visual, _ = inference(input_visual, Net_model, if_res=False)
            # output_visual = output_visual.detach().cpu()
            v_visual = output_visual[..., (0,)]**2 + output_visual[..., (1,)]**2
            coord_visual = input_visual.detach().cpu().numpy()
            field_visual = np.concatenate((v_visual, output_visual[..., (-1,)]), axis=-1)
            field_lambda = np.concatenate((lambda_Re, lambda_Im), axis=-1)
            coord_lambda = input_train.detach().cpu().numpy()[bound:]

            plt.figure(100, figsize=(15, 10))
            plt.clf()
            Visual.plot_field_horo(coord_visual, field_visual, coord_lambda, field_lambda, title=name)

            paddle.save({'epoch': iter, 'model': Net_model.state_dict(),
                         'log_loss': np.array(log_loss), 'lambda_Re': lambda_Re,
                         'lambda_Im': lambda_Im, 'mu_f': mu,},
                        os.path.join(model_path, name + 'latest_model.pdparams'))
            sio.savemat(os.path.join(data_path, name + 'langrangian.mat'),
                        {'log_loss': np.array(log_loss), 'mu_f': mu,
                         'lambda_Re': lambda_Re, 'lambda_Im': lambda_Im,
                         'coord_visual': coord_visual, 'field_visual': field_visual})


if __name__ == '__main__':

    res_path = os.path.join('res', 'horo_mu_2')
    train_path = os.path.join('res', 'horo_mu_2', 'train')
    data_path = os.path.join('res', 'horo_mu_2', 'data')
    model_path = os.path.join('res', 'horo_mu_2', 'model')
    fig_path = os.path.join('res', 'horo_mu_2', 'figure')
    isCreated = os.path.exists(res_path)
    if not isCreated:
        # os.makedirs(res_path)
        os.makedirs(train_path)
        os.makedirs(data_path)
        os.makedirs(model_path)
        os.makedirs(fig_path)

    ############################### 定义计算域，并随机生成数据 input_train ##################################
    from horo_model import l_BOX
    N = ((l_BOX[1] - l_BOX[0]) / 0.05).astype(int)  # 完整计算域
    g3_box = np.array([[-2, 0], [2, 3]])  # g3 计算objective的区域
    N_g3 = ((g3_box[1] - g3_box[0]) / 0.05).astype(int)

    # Training Data
    x_inner = gen_dataset(l_BOX, N[0]*N[1], method='Sobol', ndim=2)
    x_g3_small = gen_dataset(g3_box, N_g3[0]*N_g3[1], method='Sobol', ndim=2)
    x_boundl = np.concatenate([np.ones([N[1], 1]) * l_BOX[0, 0],
                              gen_dataset(l_BOX[:, 1], N[1], method='Sobol', ndim=1)], axis=-1)
    x_boundr = np.concatenate([np.ones([N[1], 1]) * l_BOX[1, 0],
                              gen_dataset(l_BOX[:, 1], N[1], method='Sobol', ndim=1)], axis=-1)
    x_boundd = np.concatenate([gen_dataset(l_BOX[:, 0], N[0], method='Sobol', ndim=1),
                              np.ones([N[0], 1]) * l_BOX[0, 1],], axis=-1)
    x_boundu = np.concatenate([gen_dataset(l_BOX[:, 0], N[0], method='Sobol', ndim=1),
                              np.ones([N[0], 1]) * l_BOX[1, 1],], axis=-1)

    input_train = np.concatenate([x_g3_small, x_boundl, x_boundr, x_boundd, x_boundu, x_inner], axis=0).astype(np.float32)
    num_opt = x_g3_small.shape[0]
    np.savetxt(os.path.join(train_path, 'input_train.txt'), input_train[num_opt:])

    input_train = paddle.to_tensor(input_train, dtype='float32', place='gpu:0')

    ############################### 生成用于画图的标准数据 input_visual ##################################
    x_vis = np.linspace(l_BOX[0, 0], l_BOX[1, 0], N[0]).astype(np.float32)[:, None]
    y_vis = np.linspace(l_BOX[0, 1], l_BOX[1, 1], N[1]).astype(np.float32)[:, None]

    x_vis = np.tile(x_vis, (1, y_vis.shape[0]))  # Nx x Ny
    y_vis = np.tile(y_vis, (1, x_vis.shape[0])).T  # Nx x Ny
    input_visual = np.stack((x_vis, y_vis), axis=-1)
    input_visual = paddle.to_tensor(input_visual, dtype='float32', place='gpu:0')
    input_visual.stop_gradient = False

    ############################### 定义网络模型、损失、优化器等 ##################################
    Net_model = Net(planes=[15] + [48] * 4 + [3], X=input_train)
    L2loss = nn.MSELoss()
    Optimizer1 = paddle.optimizer.Adam(learning_rate=0.001, parameters=Net_model.parameters(),
                                       beta1=0.9, beta2=0.999, epsilon=1e-8,)
    Optimizer2 = paddle.optimizer.Adam(learning_rate=0.0001, parameters=Net_model.parameters(),
                                       beta1=0.9, beta2=0.999, epsilon=1e-8,)
    Visual = visual_data.matplotlib_vision(train_path, field_name=['E', 'eps', 'lambda_Re', 'lambda_Im'], input_name=['x', 'y'])

    mu = 2.0
    Log_loss = []
    # 损失的权重： 【 PDE loss 1, PDE loss 2, Lagrangian loss 1, Lagrangian loss 2, objective loss】
    Loss_weight = [0.5 * mu] * 2 + [0.0, 0.0, 1.0]
    # 初始的朗格朗日乘子为0， 当采用拉格朗日增强方法时才进行计算
    lambla_Re, lambla_Im = np.zeros((len(input_train[num_opt:]), 1)), np.zeros((len(input_train[num_opt:]), 1))

    #################################### Soft loss ##################################
    # Adam 优化35000步
    train('Adam_Init_', 35001, input_train, num_opt, Net_model, L2loss, Loss_weight, Optimizer1,
          log_loss=Log_loss, lambda_Re=lambla_Re, lambda_Im=lambla_Im, display_epoch=1000)
    # Adam 降学习率优化，模拟 L-BGFS优化器
    train('Adam_Redc_', 5001, input_train, num_opt, Net_model, L2loss, Loss_weight, Optimizer2,
          log_loss=Log_loss, lambda_Re=lambla_Re, lambda_Im=lambla_Im, display_epoch=1000)

    #################################### Penalty ##################################
    # beta = 2
    # i = 0
    # while mu < 100:
    #     i += 1
    #     mu *= beta
    #     print("-" * 80)
    #     print(f"Iteration {i}: mu = {mu}\n")
    #     Loss_weight = [0.5 * mu] * 2 + [1, 1] + [1.0]
    #     train('Penalty_iter_' + str(i) + '_', 10001, input_train, num_opt, Net_model, L2loss, Loss_weight, Optimizer2,
    #            log_loss=Log_loss, lambda_Re=lambla_Re, lambda_Im=lambla_Im, display_epoch=1000)

    #################################### Lagrangian ##################################
    beta = 2
    for i in range(1, 10):
        input_train.stop_gradient = False
        _, residual = inference(input_train, Net_model, if_res=True)

        lambla_Re += mu * residual[num_opt:, (0,)]
        lambla_Im += mu * residual[num_opt:, (1,)]
        mu *= beta
        print("-" * 80)
        print(f"Iteration {i}: mu = {mu}\n")
        Loss_weight = [0.5 * mu] * 2 + [1, 1] + [1.0]

        train('Lag_iter_' + str(i) + '_', 10001, input_train, num_opt, Net_model, L2loss, Loss_weight, Optimizer2,
              log_loss=Log_loss, lambda_Re=lambla_Re, lambda_Im=lambla_Im, display_epoch=1000)
