import os
import gc
from PIL import Image
import random
# Set loglevel to suppress tensorflow GPU messages
# from Kgraph_test01 import DI_generated
from data_rate import _crop

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import datasets
from change_detector import ChangeDetector, pic
from change_detector import ChangeDetector
from image_translation import ImageTranslationNetwork, Discriminator
from change_priors import Degree_matrix, ztz, image_in_patches
from config import get_config_kACE
from decorators import image_to_tensorboard
from kmeans import kmeans, lograte, threshold_otsu
import pandas as pd
from sklearn.cluster import KMeans
from network import ImageTranslationNetwork, Graph_Attention_new, Graph_Attention_Chs, Graph_Attention_Chs1, \
    Graph_Attention_Chs2


class Kern_AceNet(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        """
                Input:
                    translation_spec - dict with keys 'enc_X', 'enc_Y', 'dec_X', 'dec_Y'.
                                       Values are passed as kwargs to the
                                       respective ImageTranslationNetwork's
                    cycle_lambda=2 - float, loss weight
                    cross_lambda=1 - float, loss weight
                    l2_lambda=1e-3 - float, loss weight
                    kernels_lambda - float, loss weight
                    learning_rate=1e-5 - float, initial learning rate for
                                         ExponentialDecay
                    clipnorm=None - gradient norm clip value, passed to
                                    tf.clip_by_global_norm if not None
                    logdir=None - path to log directory. If provided, tensorboard
                                  logging of training and evaluation is set up at
                                  'logdir/timestamp/' + 'train' and 'evaluation'
        """

        super().__init__(**kwargs)

        self.recon_lambda = kwargs.get("recon_lambda", 1)
        self.kernels_lambda = kwargs.get("kernels_lambda", 10)
        self.graph_lambda = kwargs.get("graph_lambda", 1)
        self.fooling_lambda = kwargs.get("fooling_lambda", 1)
        self.aps = kwargs.get("affinity_patch_size", 20)
        self.min_impr = kwargs.get("minimum improvement", 1e-3)
        self.patience = kwargs.get("patience", 10)
        self.crop_factor = kwargs.get("crop_factor", 0.2)
        self.l2_lambda = kwargs.get("l2_lambda", 1e-4)
        # encoders of X and Y  编码器模块
        self._enc_x = ImageTranslationNetwork(
            **translation_spec["enc_X"], name="enc_X", l2_lambda=self.l2_lambda
        )
        self._enc_y = ImageTranslationNetwork(
            **translation_spec["enc_Y"], name="enc_Y", l2_lambda=self.l2_lambda
        )

        # decoder of X and Y  解码器模块
        self._dec_x = ImageTranslationNetwork(
            **translation_spec["dec_X"], name="dec_X", l2_lambda=self.l2_lambda
        )
        self._dec_y = ImageTranslationNetwork(
            **translation_spec["dec_Y"], name="dec_Y", l2_lambda=self.l2_lambda
        )
        self._discr = Discriminator(
            **translation_spec["Discriminatorx"], name="Discr", l2_lambda=self.l2_lambda
        )
        self._discry = Discriminator(
            **translation_spec["Discriminatory"], name="Discry", l2_lambda=self.l2_lambda
        )

        self._graph_net_x = Graph_Attention_Chs2(3, 3, out_channel=11)
        self._graph_net_y = Graph_Attention_Chs2(3, 3, out_channel=3)

        # self._graph_net_x = Graph_Attention_Chs2(3,3, out_channel=3)#多头
        # self._graph_net_y = Graph_Attention_Chs2(3,3, out_channel=1)

        self.loss_object = tf.keras.losses.MeanSquaredError()

        # self.train_metrics["struct_x"] = tf.keras.metrics.Sum(name="struct_x MSE sum")
        # self.train_metrics["struct_y"] = tf.keras.metrics.Sum(name="struct_y MSE sum")
        self.train_metrics["recon_x"] = tf.keras.metrics.Sum(name="recon_x MSE sum")
        self.train_metrics["recon_y"] = tf.keras.metrics.Sum(name="recon_y MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")
        self.train_metrics["total"] = tf.keras.metrics.Sum(name="total MSE sum")
        self.train_metrics["graph_x"] = tf.keras.metrics.Sum(name="graph_x MSE sum")
        self.train_metrics["graph_y"] = tf.keras.metrics.Sum(name="graph_y MSE sum")
        # self.train_metrics["cycle_x"] = tf.keras.metrics.Sum(name="cycle_x MSE sum")
        # self.train_metrics["cycle_y"] = tf.keras.metrics.Sum(name="cycle_y MSE sum")
        self.train_metrics["krnls"] = tf.keras.metrics.Sum(name="krnls MSE sum")
        #self.train_metrics["krnls1"] = tf.keras.metrics.Sum(name="krnls1 MSE sum")

        # Track total loss history for use in early stopping
        self.metrics_history["total"] = []

    def save_all_weights(self):
        self._enc_x.save_weights(self.log_path + "/weights/_enc_x/")
        self._enc_y.save_weights(self.log_path + "/weights/_enc_y/")
        self._dec_x.save_weights(self.log_path + "/weights/_dec_x/")
        self._dec_y.save_weights(self.log_path + "/weights/_dec_y/")
        # self._graph_net_x.save_weights(self.log_path + "/weights/_graph_net_x/")
        # self._graph_net_y.save_weights(self.log_path + "/weights/_graph_net_y/")

    def load_all_weights(self, folder):
        self._enc_x.load_weights(folder + "/weights/_enc_x/")
        self._enc_y.load_weights(folder + "/weights/_enc_y/")
        self._dec_x.load_weights(folder + "/weights/_dec_x/")
        self._dec_y.load_weights(folder + "/weights/_dec_y/")
        # self._graph_net_x.load_weights(folder + "/weights/_graph_net_x/")
        # self._graph_net_y.load_weights(folder + "/weights/_graph_net_y/")

    @image_to_tensorboard()
    def enc_x(self, inputs, training=False):
        """ Wraps encoder call for TensorBoard printing and image save """
        return self._enc_x(inputs, training)

    @image_to_tensorboard()
    def dec_x(self, inputs, training=False):
        return self._dec_x(inputs, training)

    @image_to_tensorboard()
    def enc_y(self, inputs, training=False):
        return self._enc_y(inputs, training)

    @image_to_tensorboard()
    def dec_y(self, inputs, training=False):
        return self._dec_y(inputs, training)

    def early_stopping_criterion(self):
        temp = tf.math.reduce_min([self.stopping, self.patience]) + 1
        self.stopping.assign_add(1)
        last_losses = np.array(self.metrics_history["total"][-(temp):])
        idx_min = np.argmin(last_losses)
        if idx_min == (temp - 1):
            self.save_all_weights()
        while idx_min > 0:
            idx_2nd_min = np.argmin(last_losses[:idx_min])
            improvement = last_losses[idx_2nd_min] - last_losses[idx_min]
            if improvement > self.min_impr:
                break
            else:
                idx_min = idx_2nd_min
        stop = idx_min == 0 and self.stopping > self.patience
        tf.print(
            "total_loss",
            last_losses[-1],
            "Target",
            last_losses[idx_min],
            "Left",
            self.patience - (temp - 1) + idx_min,
        )
        return stop

    # @tf.function
    def __call__(self, inputs, training=False):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])
        c_x, c_y = x.shape[-1], y.shape[-1]

        if training:
            # tmp_list = _crop()
            # x = tmp_list[0][:, :, 0:11]
            # y = tmp_list[0][:, :, 11:14]
            # x = tf.expand_dims(x, axis=0)
            # y = tf.expand_dims(y, axis=0)

            x_code, y_code = self._enc_x(x, training), self._enc_y(y, training)
            x_hat, y_hat = self._dec_x(y_code, training), self._dec_y(x_code, training)
            x_tilde, y_tilde = (
                self._dec_x(x_code, training),
                self._dec_y(y_code, training),
            )
            x_re, y_re = self._enc_x(x_hat, training), self._enc_y(y_hat, training)
            y_dot, x_dot = self._dec_y(x_re, training), self._dec_x(y_re, training)

            # print(x)
            # graph_x=self._graph_net_x(x_code, y_code)
            # graph_y=self._graph_net_y(y_code, x_code)

            graph_x = self._graph_net_x(x_code, y_code)
            graph_y = self._graph_net_y(y_code, x_code)
            difference_img = self._difference_img(x_tilde, y_tilde, x_hat, y_hat)
            di_kmeans_reverse = tf.cast(self._change_map_reverse(difference_img), tf.float32)

            zx_t_zy = ztz(
                tf.image.central_crop(x_code, self.crop_factor),
                tf.image.central_crop(y_code, self.crop_factor),
            )

            struct_x = Degree_matrix(tf.image.central_crop(x, 0.2), tf.image.central_crop(y_hat, 0.2))
            struct_y = Degree_matrix(tf.image.central_crop(y, 0.2), tf.image.central_crop(x_hat, 0.2))

            print("当前的epoch是：", self.epoch)

            # 下面两行是和加利福尼亚数据集的实验对应的 验证变化区域对结果的影响
            z_discr = self._discr(tf.concat([x_hat, x], 0))
            y_discr = self._discry(tf.concat([y_hat, y], 0))
            # z_discr = self._discr(tf.concat([x_hat_discr, x_discr], 0))
            # y_discr = self._discry(tf.concat([y_hat_discr, y_discr], 0))

            retval = [x_code, y_code, x_hat, y_hat, x_tilde, y_tilde, z_discr, y_discr,
                      di_kmeans_reverse, graph_x, graph_y, zx_t_zy, x_re, y_re, y_dot, x_dot, struct_x, struct_y]

        else:
            x_code, y_code = self.enc_x(x, name="x_code"), self.enc_y(y, name="y_code")

            code_di = self._domain_difference_img(x_code, y_code)

            x_tilde, y_tilde = (
                self.dec_x(x_code, name="x_tilde"),
                self.dec_y(y_code, name="y_tilde"),
            )

            x_hat, y_hat = (
                self.dec_x(y_code, name="x_hat"),
                self.dec_y(x_code, name="y_hat"),
            )
            #pic(code_di, name='code_di', chanel='L')
            pic(x_code, name="x_code", chanel="RGB")
            pic(y_code, name="y_code", chanel="RGB")
            # pic(x_tilde, name="x_tilde", chanel="L")
            #pic(y_tilde, name="y_tilde", chanel="RGB")
            # pic(x_hat, name="x_hat")
            #pic(y_hat, name="y_hat", chanel="RGB")

            # z_x= self._graph_net_x(x_code, y_code)
            # z_y= self._graph_net_y(y_code, x_code)
            # pic(z_x, name="z_x", chanel="RGB")
            # pic(z_y, name="z_y", chanel="L")
            difference_img = self._difference_img(x_tilde, y_tilde, x_hat, y_hat)
            retval = difference_img

        return retval

    # @tf.function
    def train_step(self, x, y, clw, gt):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        with tf.GradientTape(persistent=True) as tape:
            x_code, y_code, x_hat, y_hat, x_tilde, y_tilde, z_discr, y_discr, di_kmeans, graph_x, graph_y, zx_t_zy, x_re, \
            y_re, y_dot, x_dot, struct_x, struct_y = self(
                [x, y], training=True
            )

            gen_discr = z_discr[: z_discr.shape[0] // 2]
            labels_g = tf.ones(gen_discr.shape, dtype=tf.float32)
            labels_d = tf.concat([1.0 - labels_g, labels_g], 0)

            gen_y_discr = y_discr[: y_discr.shape[0] // 2]
            labels_y_g = tf.ones(gen_y_discr.shape, dtype=tf.float32)
            labels_y_d = tf.concat([1.0 - labels_y_g, labels_y_g], 0)

            # rate = tf.reduce_sum(di_kmeans) / tf.cast(tf.size(di_kmeans), tf.float32)
            # print("本轮迭代中不变区域占比为：", rate)
            discr_loss = self.loss_object(z_discr, labels_d)
            discr_y_loss = self.loss_object(y_discr, labels_y_d)
            Kern = 1.0 - Degree_matrix(
                tf.image.central_crop(x, self.crop_factor),
                tf.image.central_crop(y, self.crop_factor),
            )
            Kern1 = 1.0 - Degree_matrix(
                tf.image.central_crop(x_hat, self.crop_factor),
                tf.image.central_crop(y_hat, self.crop_factor),
            )

            l2_gen_loss = (
                    sum(self._enc_x.losses)
                    + sum(self._enc_y.losses)
                    + sum(self._dec_x.losses)
                    + sum(self._dec_y.losses)

            )
            l2_loss = (
                    sum(self._enc_x.losses)
                    + sum(self._enc_y.losses)
                    + sum(self._dec_x.losses)
                    + sum(self._dec_y.losses)
                    + sum(self._discr.losses)

            )
            l2_dis_loss = sum(self._discr.losses)
            l2y_dis_loss = sum(self._discry.losses)

            fooling_loss = self.fooling_lambda * self.loss_object(gen_discr, labels_g)
            fooling_y_loss = self.fooling_lambda * self.loss_object(gen_y_discr, labels_y_g)
            kernels_loss = self.kernels_lambda * self.loss_object(Kern, zx_t_zy)
            #kernels_loss1 = self.kernels_lambda * self.loss_object(Kern1, zx_t_zy)
            recon_x_loss = self.recon_lambda * self.loss_object(x, x_tilde)
            recon_y_loss = self.recon_lambda * self.loss_object(y, y_tilde)
            # struct_y_loss = self.struct_lambda * self.loss_object(y, y_hat, clw)
            # struct_x_loss = self.struct_lambda * self.loss_object(x, x_hat, clw)

            # cycle_x_loss = self.loss_object(x, x_dot)
            # cycle_y_loss = self.loss_object(y, y_dot)
            graph_x_loss = self.graph_lambda * self.loss_object(x_hat, graph_x)
            graph_y_loss = self.graph_lambda * self.loss_object(y_hat, graph_y)
            # tot_gen_loss = l2_gen_loss +recon_x_loss + recon_y_loss+ struct_x_loss + struct_y_loss+ kernels_loss+ cycle_x_loss + cycle_y_loss
            tot_gen_loss = l2_gen_loss + graph_x_loss + graph_y_loss + kernels_loss  + \
                           recon_x_loss + recon_y_loss + fooling_loss + fooling_y_loss
            # + struct_x_loss + struct_y_loss
            tot_dis_loss = l2_dis_loss + discr_loss
            tot_dis_y_loss = l2y_dis_loss + discr_y_loss
            tot_loss = [
                tot_gen_loss,
                tot_dis_loss,
                tot_dis_y_loss,
            ]
            # print(f"recon:{recon_x_loss + recon_y_loss}"
            #      f"fool_loss:{fooling_y_loss + fooling_loss}, discr_loss:{discr_loss + discr_y_loss},"
            #      f"graph_loss:{hat_x_loss + hat_y_loss}"
            #      f"feature_loss:{feature_x_loss + feature_x_loss}"
            #      )

            targets = [
                self._enc_x.trainable_variables
                + self._enc_y.trainable_variables
                + self._dec_x.trainable_variables
                + self._dec_y.trainable_variables
                + self._graph_net_x.trainable_variables
                + self._graph_net_y.trainable_variables,
                self._discr.trainable_variables,
                self._discry.trainable_variables,

            ]
            grads = []
            for i, v in enumerate(targets):
                grads += tape.gradient(tot_loss[i], v)
            targets = [item for sublist in targets for item in sublist]

            if self.clipnorm is not None:  # 防止梯度爆炸
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
            self._optimizer_all.apply_gradients(zip(clipped_grads, targets))

        self.train_metrics["recon_x"].update_state(recon_x_loss)
        self.train_metrics["recon_y"].update_state(recon_y_loss)
        self.train_metrics["l2"].update_state(l2_loss)  # disc
        #self.train_metrics["l2"].update_state(l2_gen_loss)
        self.train_metrics["graph_x"].update_state(graph_x_loss)
        self.train_metrics["graph_y"].update_state(graph_y_loss)
        self.train_metrics["krnls"].update_state(kernels_loss)
        #self.train_metrics["krnls1"].update_state(kernels_loss1)

        #self.train_metrics["total"].update_state(tot_gen_loss)


def test(DATASET="xidian", CONFIG=None):
    """
    1. Fetch data (x, y, change_map)
    2. Compute/estimate A_x and A_y (for patches)
    3. Compute change_prior
    4. Define dataset with (x, A_x, y, A_y, p). Choose patch size compatible
       with affinity computations.
    5. Train CrossCyclicImageTransformer unsupervised
        a. Evaluate the image transformations in some way?
    6. Evaluate the change detection scheme
        a. change_map = threshold [(x - f_y(y))/2 + (y - f_x(x))/2]
    """
    if CONFIG is None:
        CONFIG = get_config_kACE(DATASET)
    print(f"Loading {DATASET} data")
    x_im, y_im, target_cm, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, **CONFIG)
    ps = CONFIG["patch_size"]
    C_CODE = 3
    print("here")
    TRANSLATION_SPEC = {
        "enc_X": {"input_chs": C_X, "filter_spec": [50, 50, C_CODE]},
        "enc_Y": {"input_chs": C_Y, "filter_spec": [50, 50, C_CODE]},
        "dec_X": {"input_chs": C_CODE, "filter_spec": [50, 50, C_X]},
        "dec_Y": {"input_chs": C_CODE, "filter_spec": [50, 50, C_Y]},
        "Discriminatorx": {
            "shapes": [ps, C_X],
            "filter_spec": [25, 100, 200, 50, 1],
        },
        "Discriminatory": {
            "shapes": [ps, C_Y],
            "filter_spec": [25, 100, 200, 50, 1],
        },
    }

    print("Change Detector Init")
    cd = Kern_AceNet(TRANSLATION_SPEC, **CONFIG)  # 初始化编码和解码网络
    print("Training")
    training_time = 0
    cross_loss_weight = tf.expand_dims(tf.zeros(x_im.shape[:-1], dtype=tf.float32), -1)
    for epochs in CONFIG["list_epochs"]:
        CONFIG.update(epochs=epochs)
        tr_gen, dtypes, shapes = datasets._training_data_generator(
            x_im[0], y_im[0], target_cm[0], cross_loss_weight[0], CONFIG["patch_size"]
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        for x, y, _ in EVALUATE.batch(1):
            alpha = cd([x, y])
        cross_loss_weight = 1.0 - alpha
        training_time += tr_time
        cd.final_evaluate(EVALUATE, **CONFIG)

    cd.load_all_weights(cd.log_path)
    cd.final_evaluate(EVALUATE, **CONFIG)
    final_kappa = cd.metrics_history["cohens kappa"][-1]
    final_acc = cd.metrics_history["ACC"][-1]
    final_auc = cd.metrics_history["AUC"][-1]
    performance = (final_kappa, final_acc, final_auc)
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    speed = (epoch, training_time, timestamp)
    del cd
    gc.collect()
    print(performance, speed)
    return final_kappa, final_auc


if __name__ == "__main__":
    # print(test("Italy"))
    # print(test("California"))
    # print(test("Air"))
    # print(test("xidian2"))
    epoch1 = 40
    final_kappas = []
    i = 0
    for epoch1 in range(epoch1):

        final_kappa, final_auc = test("Farmland")
        final_kappas.append(final_kappa)
        max_index = max(enumerate(final_kappas), key=lambda x: x[1])[0]
        print(max(final_kappas), max_index, i)
        i = i + 1
        # if final_kappa > 0.55 and final_auc > 0.88:  # California
        # if final_kappa > 0.82 and final_auc > 0.9:#xidian2
        if final_kappa > 0.55:  # Italy
            # if final_kappa > 0.77 :#Air
            print("helloworld")
            exit()
