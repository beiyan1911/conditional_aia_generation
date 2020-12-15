import os
import random
import time
from collections import deque
from astropy.io import fits
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
# from imageio import imsave
from torch.utils.tensorboard import SummaryWriter

# todo: 解决mac 保存报错，详细原因？
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from matplotlib import pyplot as plt


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6

def zscore2(im):
    im = (im - np.median(im)) / im.std()
    return im
def imnorm(im, mx=0, mi=0):
    #   图像最大最小归一化 0-1
    if mx != 0 and mi != 0:
        pass
    else:
        mi, mx = np.min(im), np.max(im)

    im2 = removenan((im - mi) / (mx - mi))

    arr1 = (im2 > 1)
    im2[arr1] = 1
    arr0 = (im2 < 0)
    im2[arr0] = 0

    return im2
def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2


def removeneg(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = im2 < 0
    im2[arr] = key

    return im2


def fitswrite(fileout, im, header=None):
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head


def tensor2np(input_image):
    """
    Converts a Tensor array into a numpy image array.
    :param input_image: the input image tensor array
    :return:
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().detach().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def mkdirs(paths):
    """
    create empty directories if they don't exist
    :param paths:  (str list) -- a list of directory paths
    :return:
    """

    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def im_auto_clip(im):
    mi = np.max([im.min(), im.mean() - 3 * im.std()])
    mx = np.min([im.max(), im.mean() + 3 * im.std()])
    return np.clip(im, mi, mx)


def make_grid(array, nrow=8, padding=2, pad_value=0, log2=False, auto_clip=False):
    assert len(np.shape(array)) == 4
    num, dim, h, w = np.shape(array)
    ncol = int(num // nrow)
    real_num = ncol * nrow
    if dim == 1:  # 灰度图
        result = np.ones((nrow * h + (nrow + 1) * padding, ncol * h + (ncol + 1) * padding), dtype=np.float32) * float(
            pad_value)

        for i in range(real_num):
            trow = i // ncol
            tcol = i % ncol

            img = array[i, 0]

            if log2:
                img = log2_img(img)

            if auto_clip:
                img = im_auto_clip(img)

            _min = np.min(img)
            _max = np.max(img)
            img = (img - _min) / (_max - _min)

            result[(trow + 1) * padding + trow * h:(trow + 1) * padding + (trow + 1) * h,
            (tcol + 1) * padding + tcol * w:(tcol + 1) * padding + (tcol + 1) * w] = img

    else:
        pass

    result = np.array(result * 255.0, dtype=np.uint8)
    return result


def write_grid_images(images, row, file_name, log2=False, auto_clip=False):
    # image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels

    result = make_grid(images, nrow=row, padding=10, pad_value=1, log2=log2, auto_clip=auto_clip)

    imsave(file_name, result)


# 预测值 和 标签 画45度线
def line_45(fake_B, real_B, title=''):
    # 标签值得最大最小值
    mi = real_B.min()
    mx = real_B.max()
    fig_valid_45 = plt.figure('valid_45')
    plt.plot(real_B.flatten(), fake_B.flatten(), '.r', MarkerSize=1)
    plt.xlim((mi, mx))
    plt.ylim((mi, mx))
    plt.plot([mi, mx], [mi, mx], '-g')
    plt.title(title)
    img_45 = figure_to_image(fig_valid_45)
    img_45 = img_45.transpose(1, 2, 0)
    plt.close()
    return img_45


def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as plt_backend_agg

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image


class SummaryHelper(object):
    def __init__(self, save_path, comment, flush_secs):
        super(SummaryHelper, self).__init__()
        self.writer = SummaryWriter(log_dir=save_path, comment=comment, flush_secs=flush_secs, filename_suffix='.log')

    def add_summary(self, current_summary, global_step):
        for key, value in current_summary.items():
            if isinstance(value, np.ndarray):
                self.writer.add_image(key, value, global_step)
            elif isinstance(value, float):
                self.writer.add_scalar(key, value, global_step)

    def add_scalar(self, key, value, global_step):
        self.writer.add_scalar(key, value, global_step)

    def add_image(self, key, value, global_step):
        self.writer.add_image(key, value, global_step)

    def add_figure(self, key, figure, global_step):
        self.writer.add_figure(key, figure, global_step=global_step)

    def add_graph(self, model, images):
        self.writer.add_graph(model, images)

    @staticmethod
    def get_current_losses(losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        # message = '(epoch: %d, iters: %d ,time: %.3f) ' % (epoch, iters, time)
        message = '  '
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        return message

    def close(self):
        self.writer.close()


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)


class Logger():

    def __init__(self, path, heads, prefixs=None, other=None):

        """
        logger
        :param path:
        :param heads:
        :param prefixs: eg.{['train','valid']}
        :param other: list，demo log names
        """
        self.path = path
        self.heads = []
        if prefixs:
            for pre in prefixs:
                self.heads.extend([pre + '_' + _ for _ in heads])
        else:
            self.heads = heads

        if other:
            self.heads.extend(other)
        self.heads.append('time')

        if not os.path.isfile(path):
            self.frames = pd.DataFrame(columns=self.heads)
            self.frames.to_csv(self.path, index=True, header=True, float_format='%.5f')
        else:
            self.frames = pd.DataFrame(columns=self.heads)

    def stack(self, loss, global_step, prefix=None):
        """
        add loss
        :param loss: dict or OrderedDict
        :param global_step: int ,the step of loss
        :return: None
        """
        assert type(global_step) == int, 'the global_step must be type int'
        idx = "%05d" % global_step
        self.frames.loc[idx, 'time'] = time.strftime("%Y-%m-%d %H:%M:%S")

        if not prefix:
            for key in loss.keys():
                self.frames.loc[idx, key] = loss[key]
        else:
            for key in loss.keys():
                self.frames.loc[idx, prefix + '_' + key] = loss[key]

    def syn2file(self):
        self.frames.to_csv(self.path, index=True, header=False, mode='a', float_format='%.5f')
        self.frames = pd.DataFrame(columns=self.heads)

    def print(self):
        print(self.frames)


class DictStack(object):
    """
    用于字典型的数据求平均值 例如 {'l1':0.5,'l2':1.8}
    """

    def __init__(self, list_names):
        from collections import OrderedDict
        self.loss_stack = OrderedDict()
        self.list_names = list_names
        self.empty()

    def empty(self):
        for name in self.list_names:
            self.loss_stack[name] = .0

    def stack(self, dict_data):
        for name in dict_data.keys():
            if isinstance(name, str):
                self.loss_stack[name] = self.loss_stack[name] + float(dict_data[name])

    def get_mean(self, count=None):
        if not count:
            return self.loss_stack
        else:
            for name in self.list_names:
                self.loss_stack[name] = self.loss_stack[name] / count
            return self.loss_stack


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def save_networks(model, save_path, cpu=True):
    if not cpu:
        # torch.save(models.module.state_dict(), save_path)
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)


def calculate_mean_loss(map_list):
    loss_dict = OrderedDict()
    for per_name in map_list[0].keys():
        tmp_list = []
        for per_map in map_list:
            tmp_list.append(float(per_map[per_name]))
        loss_dict[per_name] = np.array(tmp_list).mean()
    return loss_dict


if __name__ == '__main__':
    logger = Logger('aaa.csv', ['l1', 'l2'], other=['lr'])

    logger.stack({'l1': .5, 'l2': .7}, 7)
    logger.stack({'l1': .6, 'l2': .8}, 8)
    logger.stack({'lr': .1}, 7)
    logger.stack({'lr': .3}, 8)

    logger.syn2file()
