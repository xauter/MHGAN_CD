import os

# Set loglevel to suppress tensorflow GPU messages
# import dtcwt

# from upsample import double_linear

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
import re
from itertools import count

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from change_priors import eval_prior, remove_borders, image_in_patches

def _california(reduce=False):
    """ Load California dataset from .mat """
    mat = loadmat("data/California/UiT_HCD_California_2017.mat")

    t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (4, 4)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask

def _texas(clip=True):
    """ Load Texas dataset from .mat """
    mat = loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")

    t1 = np.array(mat["t1_L5"], dtype=np.single)
    t2 = np.array(mat["t2_ALI"], dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI_1"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]

    return t1, t2, change_mask

def _xidian(clip=True):
    t1 = plt.imread("./data/xidian/1.bmp")
    t2 = plt.imread("./data/xidian/2.bmp")
    change_mask = plt.imread("./data/xidian/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _xidian2(clip=True):
    t1 = plt.imread("./data/xidian2/1.bmp")
    t2 = plt.imread("./data/xidian2/2.bmp")
    change_mask = plt.imread("./data/xidian2/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _ychang(clip=True):
    t1 = plt.imread("./data/ychang/1.bmp")
    t2 = plt.imread("./data/ychang/2.bmp")
    change_mask = plt.imread("./data/ychang/3.bmp")
    # t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    # t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _bern(clip=True):
    t1 = plt.imread("./data/Bern/1.bmp")
    t2 = plt.imread("./data/Bern/2.bmp")
    change_mask = plt.imread("./data/Bern/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t2 = t2[:, :, 0]
    t2 = t2[..., np.newaxis]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _air(clip=True):
    t1 = plt.imread("./data/air/1.bmp")
    t2 = plt.imread("./data/air/2.bmp")
    change_mask = plt.imread("./data/air/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _square(clip=True):
    t1 = plt.imread("./data/square/1.bmp")
    t2 = plt.imread("./data/square/2.bmp")
    change_mask = plt.imread("./data/square/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _italy(clip=True):
    t1 = plt.imread("./data/Italy/1.bmp")
    t2 = plt.imread("./data/Italy/2.bmp")
    change_mask = plt.imread("./data/Italy/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask


def _Coastline(clip=True):
    t1 = plt.imread("./data/Coastline/im1.bmp")
    t2 = plt.imread("./data/Coastline/im2.bmp")
    change_mask = plt.imread("./data/Coastline/im3.bmp")
    t1 = t1[..., np.newaxis]
    t2 = t2[..., np.newaxis]
    change_mask = change_mask[..., np.newaxis]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _Ottawa(clip=True):
    t1 = plt.imread("./data/Ottawa/1997.05.bmp")
    t2 = plt.imread("./data/Ottawa/1997.08.bmp")
    change_mask = plt.imread("./data/Ottawa/reference.bmp")
    t1 = t1[:, :, 0]
    t2 = t2[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]
    t2 = t2[..., np.newaxis]
    #change_mask = change_mask[..., np.newaxis]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask, dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _Farmland(clip=True):
    t1 = plt.imread("./data/Farmland/im1.bmp")
    t2 = plt.imread("./data/Farmland/im2.bmp")
    change_mask = plt.imread("./data/Farmland/reference ps.bmp")
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]
    t2 = t2[..., np.newaxis]
    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask, dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask
def _clip(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """
    # temp = np.reshape(image, (-1, image.shape[-1]))
    #
    # limits = tf.reduce_mean(temp, 0) + 3.0 * tf.math.reduce_std(temp, 0)
    # for i, limit in enumerate(limits):
    #     channel = temp[:, i]
    #     channel = tf.clip_by_value(channel, 0, limit)
    #     ma, mi = tf.reduce_max(channel), tf.reduce_min(channel)
    #     channel = 2.0 * ((channel) / (ma)) - 1
    #     temp[:, i] = channel
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    #tf.reduce_min函数用来计算一个张量的各个维度上元素的最小值
    #tf.reduce_min(image)未指定维度时获得全局最小值
    return image


def _training_data_generator(x, y, gt, p, patch_size):
    """
        Factory for generator used to produce training dataset.
        The generator will choose a random patch and flip/rotate the images

        Input:
            x - tensor (h, w, c_x)
            y - tensor (h, w, c_y)
            p - tensor (h, w, 1)
            patch_size - int in [1, min(h,w)], the size of the square patches
                         that are extracted for each training sample.
        Output:
            to be used with tf.data.Dataset.from_generator():
                gen - generator callable yielding
                    x - tensor (ps, ps, c_x)
                    y - tensor (ps, ps, c_y)
                    p - tensor (ps, ps, 1)
                dtypes - tuple of tf.dtypes
                shapes - tuple of tf.TensorShape
    """
    gt = np.array(gt, dtype=np.float)
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    c_x, c_y, c_gt= x.shape[2], y.shape[2], gt.shape[2]
    chs = c_x + c_y + c_gt + 1
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, c_x + c_y + 1, 1)
    gt_chs = slice(c_x + c_y + 1, chs, 1)
    data = tf.concat([x, y, p, gt], axis=-1)
    #将gt放在最后一维上
    #此处的data是一个300*412*5的张量，在最后一维上将x,y,p(此处p全0.指的是交换损失权重cross_loss_weight)
    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            tmp = tf.image.rot90(tmp, np.random.randint(4))
            tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs], tmp[:, :, gt_chs]

    dtypes = (tf.float32, tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
        tf.TensorShape([patch_size, patch_size, c_gt])
    )

    return gen, dtypes, shapes


def _training_data_generator1(x, y, gt, p, patch_size):
    """
    Factory for generator used to produce training dataset.
    The generator will return patches in sequential order each time it's called,
    handling edge cases by padding the image.

    Input:
        x - tensor (h, w, c_x)
        y - tensor (h, w, c_y)
        p - tensor (h, w, 1)
        patch_size - int in [1, min(h,w)], the size of the square patches
                     that are extracted for each training sample.
    Output:
        to be used with tf.data.Dataset.from_generator():
            gen - generator callable yielding
                x - tensor (ps, ps, c_x)
                y - tensor (ps, ps, c_y)
                p - tensor (ps, ps, 1)
            dtypes - tuple of tf.dtypes
            shapes - tuple of tf.TensorShape
    """
    gt = np.array(gt, dtype=np.float32)
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    c_x, c_y, c_gt = x.shape[2], y.shape[2], gt.shape[2]
    chs = c_x + c_y + c_gt + 1  # total channels
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, c_x + c_y + 1, 1)
    gt_chs = slice(c_x + c_y + 1, chs, 1)

    # Concatenate x, y, p, and gt along the last dimension
    data = tf.concat([x, y, p, gt], axis=-1)

    # Image dimensions
    h, w = x.shape[0], x.shape[1]

    # Calculate padding required to make h and w divisible by patch_size
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    # Pad the data on the bottom and right sides
    padded_data = tf.pad(data, [[0, pad_h], [0, pad_w], [0, 0]])

    # Update the new height and width after padding
    padded_h = h + pad_h
    padded_w = w + pad_w

    # Initialize patch position
    current_i, current_j = [0], [0]

    def gen():
        # Access the current patch index through mutable list variables
        i, j = current_i[0], current_j[0]

        # Yield patch at the current position
        while i < padded_h:
            if j < padded_w:
                tmp = padded_data[i:i + patch_size, j:j + patch_size, :]
                yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs], tmp[:, :, gt_chs]
                j += patch_size  # Move to the next patch in the row
            else:
                i += patch_size  # Move to the next row
                j = 0  # Reset column index to 0 for the new row
                if i >= padded_h:
                    break

        # Update current patch position for the next call
        current_i[0], current_j[0] = i, j

    dtypes = (tf.float32, tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
        tf.TensorShape([patch_size, patch_size, c_gt])
    )

    return gen, dtypes, shapes


def _training_data_generator2(x, y, gt, p, patch_size):
    """
    Factory for generator used to produce training dataset.
    The generator will return patches in sequential order each time it's called,
    with padding added to handle edges.

    Input:
        x - tensor (h, w, c_x)
        y - tensor (h, w, c_y)
        p - tensor (h, w, 1)
        patch_size - int in [1, min(h,w)], the size of the square patches
                     that are extracted for each training sample.
    Output:
        to be used with tf.data.Dataset.from_generator():
            gen - generator callable yielding
                x - tensor (ps, ps, c_x)
                y - tensor (ps, ps, c_y)
                p - tensor (ps, ps, 1)
            dtypes - tuple of tf.dtypes
            shapes - tuple of tf.TensorShape
    """
    gt = np.array(gt, dtype=np.float32)
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    c_x, c_y, c_gt = x.shape[2], y.shape[2], gt.shape[2]
    chs = c_x + c_y + c_gt + 1  # total channels
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, c_x + c_y + 1, 1)
    gt_chs = slice(c_x + c_y + 1, chs, 1)

    # Concatenate data
    data = tf.concat([x, y, p, gt], axis=-1)

    # Image dimensions
    h, w = x.shape[0], x.shape[1]

    # Calculate the necessary padding to make dimensions divisible by patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # Pad the data
    data_padded = tf.pad(data, [[0, pad_h], [0, pad_w], [0, 0]], mode='CONSTANT')

    # Padded dimensions
    h_padded, w_padded = data_padded.shape[0], data_padded.shape[1]

    current_i, current_j = [0], [0]

    def gen():
        i, j = current_i[0], current_j[0]

        # Iterate through the padded image
        while i < h_padded:
            if j < w_padded:
                tmp = data_padded[i:i + patch_size, j:j + patch_size, :]
                yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs], tmp[:, :, gt_chs]
                j += patch_size  # Move to the next patch in the row
            else:
                i += patch_size  # Move to the next row
                j = 0  # Reset column index to 0 for the new row
                if i >= h_padded:
                    break

        # Update current patch position
        current_i[0], current_j[0] = i, j

    dtypes = (tf.float32, tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
        tf.TensorShape([patch_size, patch_size, c_gt])
    )

    return gen, dtypes, shapes


DATASETS = {
    "Texas": _texas,
    "xidian": _xidian,
    "xidian2": _xidian2,
    "California": _california,

    "Italy": _italy,
    "Square": _square,
    "Air": _air,
    "ychang": _ychang,
    "bern": _bern,
    "Coastline": _Coastline,
    "Farmland": _Farmland,
    "Ottawa": _Ottawa,
}
prepare_data = {
    "xidian": True,
    "xidian2": True,
    "Texas": True,
    "California": True,

    "Italy": True,
    "Square": True,
    "Air": True,
    "ychang": True,
    "bern": True,
    "Coastline": True,
    "Farmland": True,
    "Ottawa": True,
}


def fetch_fixed_dataset(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    try:
        initial_cm = load_prior(name, x_im.shape[:2])
    except (FileNotFoundError, KeyError) as e:
        print("Evaluating and saving prior")
        initial_cm = evaluate_prior(name, x_im, y_im, **kwargs)
    cross_loss_weight = 1 - initial_cm
    cross_loss_weight -= tf.reduce_min(cross_loss_weight)
    cross_loss_weight /= tf.reduce_max(cross_loss_weight)

    tr_gen, dtypes, shapes = _training_data_generator(
        x_im, y_im, cross_loss_weight, patch_size
    )
    training_data = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
    training_data = training_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset = [tf.expand_dims(tensor, 0) for tensor in [x_im, y_im, target_cm]]
    if not tf.config.list_physical_devices("GPU"):
        dataset = [tf.image.central_crop(tensor, 0.1) for tensor in dataset]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = shapes[0][-1], shapes[1][-1]

    return training_data, evaluation_data, (c_x, c_y)

def fetch_CGAN(name, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    ps = kwargs.get("patch_size")
    y_im, x_im, target_cm = DATASETS[name](prepare_data[name])
    if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]
    chs = [tensor.shape[-1] for tensor in dataset]
    dataset = [remove_borders(tensor, ps) for tensor in dataset]
    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))
    dataset = [image_in_patches(tensor, ps) for tensor in dataset]
    tot_patches = dataset[0].shape[0]
    return dataset[0], dataset[1], evaluation_data, (chs[0], chs[1]), tot_patches

def fetch(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    if not tf.config.list_physical_devices("GPU"):    # default: if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y ,target_cm= dataset[0], dataset[1], dataset[2]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = x_im.shape[-1], y_im.shape[-1]

    return x, y, target_cm, evaluation_data, (c_x, c_y)


if __name__ == "__main__":
    for DATASET in DATASETS:
        print(f"Loading {DATASET}")
        fetch_fixed_dataset(DATASET)
