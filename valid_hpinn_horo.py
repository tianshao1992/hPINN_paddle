import paddle
import visual_data
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from basic_model import numpy_32, tensor_32
from horo_model import Net
import seaborn as sns


def inference(inn_var, model, if_res=False):

    h = model.feature_transform(inn_var)
    w = model(h)
    f, t = model.output_transform(inn_var, w)
    if if_res:
        res = model.pde_(inn_var, h, w, t, f)
    else:
        res = model.optim_func(f.reshape([-1, 3]))

    return numpy_32(f), numpy_32(res)


if __name__ == '__main__':

    res_path = os.path.join('res', 'horo_mu_2')
    train_path = os.path.join('res', 'horo_mu_2', 'train')
    data_path = os.path.join('res', 'horo_mu_2', 'data')
    model_path = os.path.join('res', 'horo_mu_2', 'model')
    fig_path = os.path.join('res', 'horo_mu_2', 'figure')
    name = 'Lag_iter_'
    
    from horo_model import l_BOX

    ############## 定义计算域， #####################
    N = ((l_BOX[1] - l_BOX[0]) / 0.05).astype(int)
    g3_box = np.array([[-2, 0], [2, 3]])
    N_g3 = ((g3_box[1] - g3_box[0]) / 0.05).astype(int)
    num_opt = N_g3[0] * N_g3[1]

    ########################## 在g3生成验证数据 input_optim 用于计算目标函数 #########################
    x_opt = np.linspace(g3_box[0, 0], g3_box[1, 0], N_g3[0]).astype(np.float32)[:, None]
    y_opt = np.linspace(g3_box[0, 1], g3_box[1, 1], N_g3[1]).astype(np.float32)[:, None]
    x_opt = np.tile(x_opt, (1, y_opt.shape[0]))  # Nx x Ny
    y_opt = np.tile(y_opt, (1, x_opt.shape[0])).T  # Nx x Ny
    input_opt = np.stack((x_opt, y_opt), axis=-1).reshape([-1, 2])
    input_optim = paddle.to_tensor(input_opt, dtype='float32', place='gpu:0')
    input_optim.stop_gradient = False

    ##################### 在全域生成验证数据 input_valid 用于计算PDE残差 ####################
    x_val = np.linspace(l_BOX[0, 0], l_BOX[1, 0], N[0]).astype(np.float32)[:, None]
    y_val = np.linspace(l_BOX[0, 1], l_BOX[1, 1], N[1]).astype(np.float32)[:, None]
    x_val = np.tile(x_val, (1, y_val.shape[0]))  # Nx x Ny
    y_val = np.tile(y_val, (1, x_val.shape[0])).T  # Nx x Ny
    input_valid = np.stack((x_val, y_val), axis=-1).reshape([-1, 2])
    input_valid = paddle.to_tensor(input_valid, dtype='float32', place='gpu:0')
    input_valid.stop_gradient = False

    ############################## 验证集上计算目标函数 #####################
    Net_model = Net(planes=[15] + [48] * 4 + [3], X=input_optim)
    Net_model.loadmodel(os.path.join(model_path, name + '9_latest_model.pdparams'))
    output_optim, res_optim = inference(input_optim, Net_model, if_res=False)  # 计算g3区域目标函数
    print("Final optim_function: {:.4e}".format(np.mean(res_optim**2)))

    ######################## 验证集上电场输出以及PDE残差 ######################
    Net_model = Net(planes=[15] + [48] * 4 + [3], X=input_valid)
    Net_model.loadmodel(os.path.join(model_path, name + '9_latest_model.pdparams'))
    output_valid, res_valid = inference(input_valid, Net_model, if_res=True)  # 全域网格空间分布 画云图
    output_valid = output_valid.reshape([N[0], N[1], 3])
    input_valid = numpy_32(input_valid).reshape([N[0], N[1], 2])
    print("Final PDE_loss: {:.4e}".format(np.mean(res_valid**2)))

    ############################################## 获取Fig. 6 7所需的训练过程历史数据 #############################
    data = sio.loadmat(os.path.join(data_path, name + '9_langrangian.mat'))   # 最后的结果
    log_loss = data['log_loss'][:, (0, 1, 4, 5)]
    coord_lambda = np.loadtxt(os.path.join(train_path, 'input_train.txt'), delimiter=' ')    # 训练时的空间分布

    #求lambda
    lambda_Res, lambda_Ims = [], []
    for i in range(9):
        data = sio.loadmat(os.path.join(data_path, name + str(i + 1) + '_langrangian.mat'))  # 最后的结果
        lambda_Res.append(data['lambda_Re'])
        lambda_Ims.append(data['lambda_Im'])
    lambda_Res, lambda_Ims = np.array(lambda_Res).squeeze(), np.array(lambda_Ims).squeeze()

    Visual = visual_data.matplotlib_vision(fig_path, input_name=['x', 'y'],
                                           field_name=['Fig7 E', 'Fig7 eps',
                                                     'Fig 6C lambda_Re 1', 'Fig 6C lambda_Re 4', 'Fig 6C lambda_Re 9',
                                                     'Fig 6C lambda_Im 1', 'Fig 6C lambda_Im 4', 'Fig 6C lambda_Im 9'],
                                           )
    font = Visual.font

    ########################################### 训练损失 Fig 6 A #################################################
    # 从训练开始，包括软约束和拉格朗日增强方法中的各项损失
    plt.figure(300, figsize=(8, 6))
    smooth_step = 100
    vis_loss_ = log_loss[:-(log_loss.shape[0] % smooth_step), :].reshape([-1, smooth_step, log_loss.shape[1]])
    vis_loss_ = vis_loss_.mean(axis=1).squeeze()
    vis_loss = np.ones((vis_loss_.shape[0], 3), dtype=np.float32)
    vis_loss[:, 0] = vis_loss_[:, :2].mean(axis=1); vis_loss[:, 1:] = vis_loss_[:, 2:]
    for i in range(vis_loss.shape[1]):
        plt.semilogy(np.arange(vis_loss.shape[0])* smooth_step, vis_loss[:, i])
    plt.legend(['PDE loss', 'Objective loss', 'Total loss'], loc='lower left', prop=font)
    plt.xlabel('Iteration ', fontdict=font)
    plt.ylabel('Loss ', fontdict=font)
    plt.grid()
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.savefig(os.path.join(Visual.log_dir, 'valid_Loss_Fig6_A.jpg'))


    if 'Lag' in name or 'lag' in name:
        ########################################### 优化目标 Fig 6 B #################################################
        # 在拉格朗日每次迭代后的优化目标的变化
        obj = log_loss[35000:, -2][::10000]
        plt.figure(400, figsize=(10, 6))
        plt.clf()
        plt.plot(np.arange(len(obj)) + 1, obj, "bo-")
        plt.xlabel('k', fontdict=font)
        plt.ylabel('Objective', fontdict=font)
        plt.grid()
        plt.yticks(fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.savefig(os.path.join(Visual.log_dir, 'valid_Objetive_Fig6_B.jpg'))

        ########################################### Lagrange乘子 Fig 6 C #################################################
        iter_ind = [0, 3, 8]
        field_lambda = np.concatenate([lambda_Res[iter_ind], lambda_Ims[iter_ind]], axis=0).T
        v_visual = output_valid[..., 0] ** 2 + output_valid[..., 1] ** 2
        field_visual = np.stack((v_visual, output_valid[..., -1]), axis=-1)
        coord_visual = input_valid
        Visual.plot_field_horo(coord_visual, field_visual, coord_lambda, field_lambda, title='valid_')

        ############################################# lambda/mu #####################################################
        mu_ = 2 ** np.arange(2, 11)
        lambda_Res, lambda_Ims = lambda_Res/mu_[:, None], lambda_Ims/mu_[:, None]

        ########################################### lambda/mu Fig 6 D #################################################
        # 随机挑选3个点 表示 lambdaRe lambdaIm 的迭代1-9的过程， 结论：lambda 收敛
        np.random.seed(555)
        ind = np.random.randint(low=0, high=lambda_Res.shape[-1], size=3)
        la_mu_ind = [lambda_Res[:, ind], lambda_Ims[:, ind]]
        marker = ['ro-', 'bo:', 'r*-', 'b*:', 'rp-', 'bp:', ]
        plt.figure(500, figsize=(7, 5))
        plt.clf()
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
        for i in range(6):
            plt.plot(np.arange(1, 10), la_mu_ind[int(i % 2)][:, int(i/2)], marker[i], linewidth=2)
        plt.legend(['Re, 1', 'Im, 1', 'Re, 2', 'Im, 2', 'Re, 3', 'Im, 3', ], loc='upper right', prop=font)
        plt.grid()
        plt.xlabel('k', fontdict=font)
        plt.ylabel(r'$ \lambda^k / \mu^k_F$', fontdict=font)
        plt.yticks(fontproperties='Times New Roman', size=12)
        plt.xticks(fontproperties='Times New Roman', size=12)
        plt.savefig(os.path.join(Visual.log_dir,'valid_lambda_Fig6_D.jpg'))

        ########################################### lamda/mu Fig 6E & Fig 6F #################################################
        # 5个迭代步中的lambda 分布情况
        iter_ind = [0, 3, 5, 8]  # 即1th, 4th, 6th, 9th 迭代
        la_mu = [lambda_Res[iter_ind, :], lambda_Ims[iter_ind, :]]
        plt.figure(600, figsize=(5, 5))
        plt.clf()
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
        for i in range(len(iter_ind)):
            sns.kdeplot(la_mu[0][i, :], label='k = ' + str(iter_ind[i] + 1), cut=0, linewidth=2)
        plt.legend(prop=font)
        plt.grid()
        plt.xlim([-0.1, 0.1])
        plt.xlabel(r'$ \lambda^k_{Re} / \mu^k_F$', fontdict=font)
        plt.ylabel('Frequency', fontdict=font)
        plt.yticks(fontproperties='Times New Roman', size=12)
        plt.xticks(fontproperties='Times New Roman', size=12)
        plt.savefig(os.path.join(Visual.log_dir,'valid_lambda_Fig6_E.jpg'))

        plt.figure(700, figsize=(5, 5))
        plt.clf()
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
        for i in range(len(iter_ind)):
            sns.kdeplot(la_mu[1][i, :], label='k = ' + str(iter_ind[i] + 1), cut=0, linewidth=2)
        plt.legend(prop=font)
        plt.grid()
        plt.xlim([-0.1, 0.1])
        plt.xlabel(r'$ \lambda^k_{Im} / \mu^k_F$', fontdict=font)
        plt.ylabel('Frequency', fontdict=font)
        # plt.rcParams['font.size'] = 2
        plt.yticks(fontproperties='Times New Roman', size=12)
        plt.xticks(fontproperties='Times New Roman', size=12)
        plt.savefig(os.path.join(Visual.log_dir,'valid_lambda_Fig6_F.jpg'))





