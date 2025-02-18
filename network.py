from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import Model
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def calculate_change_alignment_loss(BDM, FDM):
    """
    计算变化对齐损失函数

    参数:
    BDM -- 反向差异图 (TensorFlow tensor)
    FDM -- 正向差异图 (TensorFlow tensor)

    返回:
    L_ca -- 变化对齐损失
    """
    # 计算每个像素位置的点积
    dot_product = tf.multiply(BDM, FDM)
    # 计算总和
    sum_dot_product = tf.reduce_sum(dot_product)
    # 图像的总像素数
    total_pixels = BDM.shape[0] * BDM.shape[1]
    # 计算变化对齐损失
    L_ca = -1 / total_pixels * sum_dot_product
    return L_ca

class downnet(Model):
    def __init__(
            self,
            dim,
            leaky_alpha=0.3
    ):
        super(downnet, self).__init__()
        self.leaky_alpha = leaky_alpha
        self.conv = Conv2D(filters=10,
                           input_shape=(None, None, dim),
                           kernel_size=1,
                           padding="valid",
                           kernel_initializer="GlorotNormal")
        self.pool = MaxPooling2D(pool_size=2, strides=2)
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = relu(x, alpha=self.leaky_alpha)
        x = self.pool(x)
        return x

class upnet(Model):
    def __init__(self,
                 leaky_alpha=0.3):
        super(upnet, self).__init__()
        self.upconv = UpSampling2D(size=(2, 2))
        self.leaky_alpha=leaky_alpha
    def call(self, inputs, training=False):
        x = self.upconv(inputs)
        x = relu(x, alpha=self.leaky_alpha)
        return x

class ImageTranslationNetwork(Model):
    """
        Same as network in Luigis cycle_prior.

        Not supporting discriminator / Fully connected output.
        Support for this should be implemented as a separate class.
    """

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)

        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            # "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Conv2D(
            filter_spec[0],
            input_shape=(None, None, input_chs),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Conv2D(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        return tanh(x)

class preshareNetwork(Model):
    """
        Same as network in Luigis cycle_prior.

        Not supporting discriminator / Fully connected output.
        Support for this should be implemented as a separate class.
    """

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)

        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            # "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Conv2D(
            filter_spec[0],
            input_shape=(None, None, input_chs),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Conv2D(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        return relu(x, alpha=self.leaky_alpha)

class Siamesenet(Model):
    def __init__(self):
        super(Siamesenet, self).__init__()
        self.conv1 = Conv2D(filters=50, input_shape=(None, None, 3), kernel_initializer="GlorotNormal", kernel_regularizer=l2(1e-3), kernel_size=3, strides=1, padding='same')
        self.conv2 = Conv2D(filters=50, kernel_initializer="GlorotNormal", kernel_regularizer=l2(1e-3), kernel_size=3, strides=1, padding='same')
        self.conv3 = Conv2D(filters=50, kernel_initializer="GlorotNormal", kernel_regularizer=l2(1e-3), kernel_size=3, strides=1, padding='same')
        self.conv4 = Conv2D(filters=3, kernel_initializer="GlorotNormal", kernel_regularizer=l2(1e-3), kernel_size=3, strides=1, padding='same')
    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = relu(x, alpha=0.3)
        x = self.conv2(x)
        x = relu(x, alpha=0.3)
        x = self.conv3(x)
        x = relu(x, alpha=0.3)
        x = self.conv4(x)
        return tanh(x)

class Graph_Attention_Union(Model):
    def __init__(self,
                 in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()
        #search region nodes linear transformation
        self.support = Conv2D(filters=in_channel, kernel_size=1)

        #target template nodes linear transformation
        self.query = Conv2D(filters=in_channel, kernel_size=1)

        self.g = Sequential([
            Conv2D(filters=in_channel, kernel_size=1),
            BatchNormalization(in_channel),
            ReLU()
        ])
        self.fi = Sequential([
            Conv2D(out_channel, input_shape=(None, None, in_channel*2), kernel_size=1),
            BatchNormalization(out_channel),
            ReLU()
        ])
    def call(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)
        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        # zf_train_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_train_plain = tf.reshape(zf_trans, [-1, shape_z[1] * shape_z[2], shape_z[3]])
        zf_train_plain = tf.transpose(zf_train_plain, perm=[0, 2, 1])
        # zf_g_plain = zf_g.view(-1, shape_z[2] * shape_z[3]).permute(0, 2, 1)
        zf_g_plain = tf.reshape(zf_g, [-1, shape_z[1] * shape_z[2], shape_z[3]])
        # xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        xf_trans_plain = tf.reshape(xf_trans, [-1, shape_x[1] * shape_x[2], shape_x[3]])

        similar = tf.matmul(xf_trans_plain, zf_train_plain)
        similar = softmax(similar, axis=2)

        # embedding = tf.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = tf.matmul(similar, zf_g_plain)
        # embedding = tf.transpose(embedding, perm=[0, 2, 1])
        # embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])
        embedding = tf.reshape(embedding, [-1, shape_x[1], shape_x[2], shape_x[3]])

        #aggregated feature
        output = tf.concat([embedding, xf_g], 3)
        output = self.fi(output)
        return output

class Graph_Attention_new(Model):
    def __init__(self,
                 in_channel, out_channel):
        super(Graph_Attention_new, self).__init__()
        #search region nodes linear transformation
        # self.conv_down = Conv2D(filters=in_channel, kernel_size=5, strides=5)
        # self.conv_up = Conv2DTranspose(filters= out_channel, kernel_size=5, strides=5)
        self.support = Conv2D(filters=in_channel, kernel_size=2, strides=2)

        #target template nodes linear transformation
        self.query = Conv2D(filters=in_channel, kernel_size=2, strides=2)

        self.g = Sequential([
            Conv2D(filters=in_channel, kernel_size=2, strides=2),
            BatchNormalization(in_channel),
            ReLU()
        ])
        self.fi = Sequential([
            Conv2DTranspose(input_shape=(None, None, in_channel*2),filters=out_channel, kernel_size=2, strides=2),
            # Conv2D(out_channel, input_shape=(None, None, in_channel*2), kernel_size=1),
            BatchNormalization(out_channel),
            ReLU()
        ])
    def call(self, zf, xf, training=True):
        # linear transformation
        # zf = self.conv_down(zf)
        # xf = self.conv_down(xf)
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)
        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        # zf_train_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_train_plain = tf.reshape(zf_trans, [-1, shape_z[1] * shape_z[2], shape_z[3]])
        zf_train_plain = tf.transpose(zf_train_plain, perm=[0, 2, 1])
        # zf_g_plain = zf_g.view(-1, shape_z[2] * shape_z[3]).permute(0, 2, 1)
        zf_g_plain = tf.reshape(zf_g, [-1, shape_z[1] * shape_z[2], shape_z[3]])
        # xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        xf_trans_plain = tf.reshape(xf_trans, [-1, shape_x[1] * shape_x[2], shape_x[3]])

        similar = tf.matmul(xf_trans_plain, zf_train_plain)
        similar = softmax(similar, axis=2)

        # embedding = tf.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = tf.matmul(similar, zf_g_plain)
        # embedding = tf.transpose(embedding, perm=[0, 2, 1])
        # embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])
        embedding = tf.reshape(embedding, [-1, shape_x[1], shape_x[2], shape_x[3]])

        #aggregated feature
        output = tf.concat([embedding, xf_g], 3)
        output = self.fi(output)
        return output


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Graph_Attention_Chs(keras.Model):
    def __init__(self, input_chs_x, input_chs_z, out_channel):
        super(Graph_Attention_Chs, self).__init__()

        # Search region nodes linear transformation
        self.support = layers.Conv2D(filters=input_chs_z, kernel_size=5, strides=5)

        # Target template nodes linear transformation
        self.query = layers.Conv2D(filters=input_chs_x, kernel_size=5, strides=5)

        self.g_x = keras.Sequential([
            layers.Conv2D(filters=input_chs_x, kernel_size=5, strides=5),
            layers.BatchNormalization(input_chs_x),
            layers.ReLU()
        ])

        self.g_z = keras.Sequential([
            layers.Conv2D(filters=input_chs_z, kernel_size=5, strides=5),
            layers.BatchNormalization(input_chs_z),
            layers.ReLU()
        ])

        self.fi = keras.Sequential([
            layers.Conv2DTranspose(filters=out_channel, kernel_size=5, strides=5),
            layers.BatchNormalization(out_channel),
            layers.ReLU()
        ])

    def call(self, zf, xf, training=True):
        # Linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # Linear transformation for message passing
        xf_g = self.g_x(xf)
        zf_g = self.g_z(zf)

        # Calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_train_plain = tf.reshape(zf_trans, [-1, shape_z[1] * shape_z[2], shape_z[3]])
        zf_train_plain = tf.transpose(zf_train_plain, perm=[0, 2, 1])

        zf_g_plain = tf.reshape(zf_g, [-1, shape_z[1] * shape_z[2], shape_z[3]])

        xf_trans_plain = tf.reshape(xf_trans, [-1, shape_x[1] * shape_x[2], shape_x[3]])

        similar = tf.matmul(xf_trans_plain, zf_train_plain)
        similar = tf.nn.softmax(similar, axis=2)

        embedding = tf.matmul(similar, zf_g_plain)
        embedding = tf.reshape(embedding, [-1, shape_x[1], shape_x[2], shape_x[3]])

        # Aggregated feature
        output = tf.concat([embedding, xf_g], axis=-1)  # Axis -1 is the channel axis
        output = self.fi(output)

        return output
import tensorflow as tf
from tensorflow.keras import layers, Model

class Graph_Attention_Chs1(keras.Model):
    def __init__(self, input_chs_x, input_chs_z, out_channel):
        super(Graph_Attention_Chs1, self).__init__()

        # Search region nodes linear transformation (input_chs_z channels to input_chs_z channels)
        self.support = layers.Conv2D(filters=input_chs_z, kernel_size=3, strides=1, padding='same')

        # Target template nodes linear transformation (input_chs_x channels to input_chs_x channels)
        self.query = layers.Conv2D(filters=input_chs_x, kernel_size=3, strides=1, padding='same')

        # Feature extractor for xf (input_chs_x channels to input_chs_x channels)
        self.g_x = keras.Sequential([
            layers.Conv2D(filters=input_chs_x, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Feature extractor for zf (input_chs_z channels to input_chs_z channels)
        self.g_z = keras.Sequential([
            layers.Conv2D(filters=input_chs_z, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Output feature generator (Concatenated channels to out_channel channels)
        self.fi = keras.Sequential([
            layers.Conv2DTranspose(filters=out_channel, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, zf, xf, training=True):
        # Linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # Linear transformation for message passing
        xf_g = self.g_x(xf)
        zf_g = self.g_z(zf)

        # Calculate similarity
        shape_x = tf.shape(xf_trans)
        shape_z = tf.shape(zf_trans)

        zf_trans_plain = tf.reshape(zf_trans, [-1, shape_z[1] * shape_z[2], shape_z[3]])
        zf_trans_plain = tf.transpose(zf_trans_plain, perm=[0, 2, 1])

        zf_g_plain = tf.reshape(zf_g, [-1, shape_z[1] * shape_z[2], shape_z[3]])

        xf_trans_plain = tf.reshape(xf_trans, [-1, shape_x[1] * shape_x[2], shape_x[3]])

        # Similarity matrix and embedding
        similar = tf.matmul(xf_trans_plain, zf_trans_plain)
        similar = tf.nn.softmax(similar, axis=2)

        embedding = tf.matmul(similar, zf_g_plain)
        embedding = tf.reshape(embedding, [-1, shape_x[1], shape_x[2], shape_x[3]])

        # Aggregated feature
        output = tf.concat([embedding, xf_g], axis=-1)  # Axis -1 is the channel axis
        output = self.fi(output)

        return output

class Graph_Attention_Chs2(Model):
    def __init__(self, input_chs_x, input_chs_z, out_channel, num_heads=4):
        super(Graph_Attention_Chs2, self).__init__()

        # 多头注意力的数量
        self.num_heads = num_heads

        # Search region nodes linear transformation (input_chs_z channels to input_chs_z channels)
        self.support = layers.Conv2D(filters=input_chs_z, kernel_size=5, strides=2, padding='same')#线性变换

        # Target template nodes linear transformation (input_chs_x channels to input_chs_x channels)
        self.query = layers.Conv2D(filters=input_chs_x, kernel_size=5, strides=2, padding='same')#线性变换

        # Feature extractor for xf (input_chs_x channels to input_chs_x channels)
        self.g_x = keras.Sequential([
            layers.Conv2D(filters=input_chs_x, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Feature extractor for zf (input_chs_z channels to input_chs_z channels)
        self.g_z = keras.Sequential([
            layers.Conv2D(filters=input_chs_z, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        # Output feature generator (Concatenated channels to out_channel channels)
        self.fi = keras.Sequential([
            layers.Conv2DTranspose(filters=out_channel, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def build_graph_attention(self, xf_trans, zf_trans, zf_g):
        """多头图注意力机制."""
        heads = []
        for _ in range(self.num_heads):
            # 计算每个头的相似度
            zf_trans_head = tf.reshape(zf_trans, [-1, tf.shape(zf_trans)[1] * tf.shape(zf_trans)[2], tf.shape(zf_trans)[3]])
            zf_trans_head = tf.transpose(zf_trans_head, perm=[0, 2, 1])
            xf_trans_head = tf.reshape(xf_trans, [-1, tf.shape(xf_trans)[1] * tf.shape(xf_trans)[2], tf.shape(xf_trans)[3]])

            similarity = tf.matmul(xf_trans_head, zf_trans_head)
            similarity = tf.nn.softmax(similarity, axis=2)#点积运算，生成相似度矩阵，然后通过 softmax 进行归一化

            # 使用相似度进行特征聚合
            zf_g_plain = tf.reshape(zf_g, [-1, tf.shape(zf_g)[1] * tf.shape(zf_g)[2], tf.shape(zf_g)[3]])
            embedding = tf.matmul(similarity, zf_g_plain)#将搜索区域图像的特征聚合到模板图像的特征上。模板图像的特征会被搜索区域的特征增强
            embedding = tf.reshape(embedding, [-1, tf.shape(xf_trans)[1], tf.shape(xf_trans)[2], tf.shape(xf_trans)[3]])

            heads.append(embedding)

        # 将多个头的输出拼接在一起
        multi_head_output = tf.concat(heads, axis=-1)
        return multi_head_output

    def call(self, zf, xf, training=True):
        # Linear transformation
        xf_trans = self.query(xf)  # 转换目标模板节点特征，将输入图像特征映射到一个新的特征空间，方便后续注意力机制的计算。
        zf_trans = self.support(zf)  # 转换搜索区域特征

        # Linear transformation for message passing
        xf_g = self.g_x(xf)  # 对模板节点特征进一步提取，用于从提取更高层次的特征。由一个卷积层、批归一化层和ReLU激活函数组成，能够提取更复杂的特征。
        zf_g = self.g_z(zf)  # 对搜索区域特征进一步提取

        # 使用多头图注意力机制来聚合特征
        embedding = self.build_graph_attention(xf_trans, zf_trans, zf_g)#融合x的z特征，生成增强后的特征 embedding。

        # Aggregated feature (拼接后通过卷积转置)
        output = tf.concat([embedding, xf_g], axis=-1)
        output = self.fi(output)

        return output



if __name__=="__main__":
    # net = Siamesenet()
    # net.build(input_shape=(1, 512, 512, 30))
    # print(net.summary())
    a = tf.random.normal([1, 500, 500, 3], mean=0, stddev=0.1)
    b = tf.random.normal([1, 500, 500, 3], mean=0, stddev=0.1)
    attention = Graph_Attention_new(in_channel=3, out_channel=3)
    c = attention(a, b)
    print(c.shape)








