
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
# import pandas as pd
import math

'''-----------定义两图片取对数函数----------------------------------------------------------------'''

def lograte(img1,img2):
    [rows, cols, chls] = img1.shape           #把图片像素的行数，列数,通道数返回给rows，cols，chls
    rate_img = np.zeros([rows, cols, chls])   #初始化矩阵
    log_img = np.zeros([rows, cols, chls])
    for i in range(rows):                     #遍历每一个像素
        for j in range(cols):
            for k in range(chls):
                rate_img[i, j, k] = np.true_divide(img1[i, j, k]+0.1, img2[i, j, k]+0.1)
                #两图像做除后将数据追加到初始化矩阵中-------->>为何要作除
                log_img[i, j, k] = abs(math.log(rate_img[i, j, k]))
                #差异图取对数后将数据追加到初始化矩阵中------------->>为何取对数
    return log_img                           #得到取对数后的变化检测差异图（矩阵）


'''--------调用K-means算法对提取到的变化信息分类并显示图片-----------------------------------------'''
'''需要完成任务：
①初始聚类中心的选择（2个）
②点与聚类中心的距离计算,后按最小距离聚类
③更新点与聚类中心距离和最小点
'''
def kmeans(log_img):
    # log_img = lograte(img1, img2)
    # log_img = (log_img * 255).astype(np.double)
    [rows, cols, chls] = log_img.shape                  #初始化矩阵
    log_img_one = log_img[:,:,0]                        #通道数置为0
    pre_img = log_img_one.reshape((rows * cols, 1))     #压缩维度，变为rows * cols行，1列
    k=2                                                 #聚类中心个数
    list_img = []                                       #初始聚类中心列表
    while len(list_img) < k:                            #只选两个点后跳出
    #while list_img.__len__() < k:                      #__len__()可以返回容器中元素的个数.len是获取集合长度的函数,它通过调用对象的__len__方法来工作
        n = np.random.randint(0,  pre_img.shape[0],  1) #以1为步长，随机选取0到行数间的数据 .shape[0]为图像高度（矩阵行数）[1]为宽度（列数）[2]为通道数
        if n not in list_img:
            list_img.append(n[0])                       #选取初始聚类中
    pre_point = pre_img[np.array(list_img)]             #初始图像的初始聚类中心点（矩阵形式）


#①定义一个空列表
#②while循环，当列表里的个数等于给定的聚类中心时停止
#③从0到行数值之间以1为步长，随机选数字
#④如果不再列表中则加入
#⑤找到初始聚类点，跳出循环
#⑥至此，完成了初始聚类中心的选择
    c=0                       #定义迭代次数
    while True:                                         #一直循环，知道遇到break跳出
        distance = [np.sum(np.sqrt((pre_img - i) ** 2)   , axis=1)   for i in pre_point]#计算每个点与两聚类中心的距离之和
        now_point = np.argmin(distance, axis=0)         #距离之和最小的点为聚类中心（返回索引）
        now_piont_distance = np.array(  list([np.average(pre_img[now_point == i], axis=0) for i in range(k)])  )

        c+=0
        if np.sum(now_piont_distance - pre_point) < 1e-10 or c>200:
            break
        else:
            pre_point = now_piont_distance
#①欧氏距离的计算
#②将距离最小点选出
#③计算新选点距离，方便与前聚类中心比较
#④若聚类中心不再变化或者迭代次数大于50次跳出
#⑤差异信息的分类完成
    labels=now_point
    res = labels.reshape((rows, cols))
    ima = np.zeros([rows, cols, chls])   #初始化聚类中心
    ima[:, :, 0] = res
    ima[:, :, 1] = res
    ima[:, :, 2] = res

    #恢复三通道
    plt.title("K-means-Change Detection Results")
    plt.imshow(ima)
    plt.axis('off')
    plt.show()
    return ima

if __name__ == '__main__':
    img1 = plt.imread('./1.bmp')
    img2 = plt.imread('./2.bmp')
    dim = lograte(img1,img2)
    dim = (dim*255).astype(np.double)

    # plt.imshow(dim)
    # plt.title("LR DI")
    # plt.show()
    kmeans(dim)


