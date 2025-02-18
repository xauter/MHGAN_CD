
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
# import pandas as pd
import math



def lograte(img1,img2):
    img1 = img1[0].numpy()
    img2 = img2[0].numpy()
    [rows, cols, chls] = img1.shape
    rate_img = np.zeros([rows, cols, chls])
    log_img = np.zeros([rows, cols, chls])
    # base_number = abs(min(np.min(img1), np.min(img2))) + 0.1
    for i in range(rows):
        for j in range(cols):
            for k in range(chls):
                rate_img[i, j, k] = abs(np.true_divide(img1[i, j, k]+0.00001, img2[i, j, k]+0.00001))

                # log_img[i, j, k] = math.log(rate_img[i, j, k])
    return rate_img

 # def _domain_difference_img(
 #        self, original, transformed, bandwidth=tf.constant(3, dtype=tf.float32)
 #    ):
 #        """
 #            Compute difference image in one domain between original image
 #            in that domain and the transformed image from the other domain.
 #            Bandwidth governs the norm difference clipping threshold
 #        """
 #        d = tf.norm(original - transformed, ord=2, axis=-1)
 #        # threshold = tf.math.reduce_mean(d) + bandwidth * tf.math.reduce_std(d)
 #        # d = tf.where(d < threshold, d, threshold)
 #        d = (d - tf.reduce_min(d)) / (tf.reduce_max(d) - tf.reduce_min(d))
 #        return tf.expand_dims(d, -1)
 #        # return tf.expand_dims(d / tf.reduce_max(d), -1)


def kmeans(img1,img2):
    # plt.subplot(2, 2, 1)
    # plt.imshow(img1[0])
    # plt.title("X_enc")
    #
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(img2[0])
    # plt.title("Y_enc")

    log_img = lograte(img1, img2)
    # plt.subplot(2, 2, 3)
    # plt.imshow(log_img)
    # plt.title("LR DI")


    [rows, cols, chls] = log_img.shape
    log_img_one = log_img[:,:,0]
    pre_img = log_img_one.reshape((rows * cols, 1))
    k=2
    list_img = []
    while len(list_img) < k:
    #while list_img.__len__() < k:
        n = np.random.randint(0,  pre_img.shape[0],  1)
        if n not in list_img:
            list_img.append(n[0])
    pre_point = np.array([np.min(pre_img),np.max(pre_img)])
    c=0
    while True:
        distance = [np.sum(np.sqrt((pre_img - i) ** 2)   , axis=1)   for i in pre_point]
        now_point = np.argmin(distance, axis=0)
        now_piont_distance = np.array(  list([np.average(pre_img[now_point == i], axis=0) for i in range(k)])  )
        c+=0
        if np.sum(now_piont_distance - pre_point) < 1e-7 or c>50:
        # if c>50:
            break
        else:
            pre_point = now_piont_distance

    labels=now_point
    res = labels.reshape((rows, cols))
    ima = np.zeros([rows, cols, chls])
    ima[:, :, 0] = res
    # ima[:, :, 1] = res
    # ima[:, :, 2] = res
    # plt.subplot(2, 2, 4)
    # plt.imshow(ima)
    # plt.title("K-means-Change DI")
    # plt.show()
    return ima

def threshold_otsu(image):
    """Return threshold value based on Otsu's method. Adapted to tf from sklearn
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If ``image`` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = (
            "threshold_otsu is expected to work correctly only for "
            "grayscale images; image shape {0} looks like an RGB image"
        )
        # warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    tf.debugging.assert_none_equal(
        tf.math.reduce_min(image),
        tf.math.reduce_max(image),
        summarize=1,
        message="expects more than one image value",
    )

    hist = tf.histogram_fixed_width(image, tf.constant([0, 255]), 256)
    hist = tf.cast(hist, tf.float32)
    bin_centers = tf.range(0.5, 256, dtype=tf.float32)

    # class probabilities for all possible thresholds
    weight1 = tf.cumsum(hist)
    weight2 = tf.cumsum(hist, reverse=True)
    # class means for all possible thresholds
    mean = tf.math.multiply(hist, bin_centers)
    mean1 = tf.math.divide(tf.cumsum(mean), weight1)
    # mean2 = (tf.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    mean2 = tf.math.divide(tf.cumsum(mean, reverse=True), weight2)

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    tmp1 = tf.math.multiply(weight1[:-1], weight2[1:])
    tmp2 = (mean1[:-1] - mean2[1:]) ** 2
    variance12 = tf.math.multiply(tmp1, tmp2)

    idx = tf.math.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


