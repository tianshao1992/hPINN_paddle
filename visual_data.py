import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import ticker
import os


class matplotlib_vision(object):

    def __init__(self, log_dir, input_name=('x'), field_name=('f',)):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()

        self.field_name = field_name
        self.input_name = input_name
        self._cbs = [None] * len(self.field_name) * 3

        # gs = gridspec.GridSpec(1, 1)
        # gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        # gs_dict = {key: value for key, value in gs.__dict__.items() if key in gs._AllowedKeys}
        # self.fig, self.axes = plt.subplots(len(self.field_name), 3, gridspec_kw=gs_dict, num=100, figsize=(30, 20))
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}


    def plot_loss(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="lower left", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('loss value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)


    def plot_field_horo(self, coord_visual, field_visual, coord_lambda, field_lambda, title=None):

        fmin, fmax = np.array([0, 1.0]), np.array([0.6, 12])
        cmin, cmax = coord_visual.min(axis=(0, 1)), coord_visual.max(axis=(0, 1))
        emin, emax = np.array([-3, -1]), np.array([3, 0])
        x_pos = coord_visual[:, :, 0]
        y_pos = coord_visual[:, :, 1]

        for fi in range(len(self.field_name)):
            ########      Exact f(t,x,y)     ###########
            # plt.subplot(1, Num_fields,  0 * Num_fields + fi + 1)
            # plt.contour(x_pos, y_pos, f_true, levels=20, linestyles='-', linewidths=0.4, colors='k')
            if fi == 0:
                plt.figure(101, figsize=(8, 6))
                plt.clf()
                plt.rcParams['font.size'] = 20
                f_true = field_visual[..., fi]
                plt.pcolormesh(x_pos, y_pos, f_true, cmap='rainbow', shading='gouraud', antialiased=True, snap=True)
                cb = plt.colorbar()
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
                plt.clim(vmin=fmin[fi], vmax=fmax[fi])
                # plt.axis('equal')
            elif fi == 1:
                plt.figure(201, figsize=(8, 1.5))
                plt.clf()
                plt.rcParams['font.size'] = 20
                f_true = field_visual[..., fi]
                plt.pcolormesh(x_pos, y_pos, f_true, cmap='rainbow', shading='gouraud', antialiased=True, snap=True)
                cb = plt.colorbar()
                # plt.axis('equal')
                plt.axis((emin[0], emax[0], emin[1], emax[1]))
                plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            else:
                plt.figure(fi * 100 + 101, figsize=(8, 6))
                plt.clf()
                plt.rcParams['font.size'] = 20
                f_true = field_lambda[..., fi - field_visual.shape[-1]]
                plt.scatter(coord_lambda[..., 0], coord_lambda[..., 1], c=f_true, cmap='rainbow', alpha=0.6)
                cb = plt.colorbar()
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
                #

            # cb.set_label('$' + self.field_name[fi] + '$', rotation=0, fontdict=self.font, y=1.12)
            # 设置图例字体和大小
            cb.ax.tick_params(labelsize=30)
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_family('Times New Roman')
            tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
            cb.locator = tick_locator
            cb.update_ticks()

            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            plt.savefig(os.path.join(self.log_dir, title + str(self.field_name[fi]) + '.jpg'))

    def plot_field_stokes(self, coord_visual, field_visual, title=None):

        fmin, fmax = field_visual.min(axis=(0, 1)), field_visual.max(axis=(0, 1))
        cmin, cmax = coord_visual.min(axis=(0, 1)), coord_visual.max(axis=(0, 1))
        x_pos = coord_visual[:, :, 0]
        y_pos = coord_visual[:, :, 1]
        Num_fields = field_visual.shape[-1]

        cmps = ['RdBu_r', 'RdBu_r', 'gray', 'gray']
        # font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        for fi in range(Num_fields):
            plt.figure(fi + 1, figsize=(15, 10))
            plt.clf()
            plt.rcParams['font.size'] = 20
            ########      Exact f(t,x,y)     ###########
            # plt.subplot(1, Num_fields,  0 * Num_fields + fi + 1)
            f_true = field_visual[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_true, cmap=cmps[fi], shading='gouraud', antialiased=True, snap=True)
            cb = plt.colorbar()
            # plt.contour(x_pos, y_pos, f_true, levels=20, linestyles='-', linewidths=0.4, colors='k')
            plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            plt.title(self.field_name[fi], font=self.font, fontsize=30)
            # 设置图例字体和大小
            cb.ax.tick_params(labelsize=20)
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_family('Times New Roman')
            tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
            cb.locator = tick_locator
            cb.update_ticks()
            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            plt.savefig(os.path.join(self.log_dir, title + str(self.field_name[fi]) + '.jpg'))