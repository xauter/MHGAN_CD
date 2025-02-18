import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

# 读取change_map和ground_truth图像
change_map = np.array(Image.open(r'D:\zhaoqin\BAACL\logs\Coastline\20250115-162319\images\2_Coastline.bmp').convert('L'))  # 灰度图
ground_truth = np.array(Image.open(r'D:\zhaoqin\BAACL\data\Coastline\im3.bmp').convert('L'))  # 灰度图

# 确保两幅图像大小一致
if change_map.shape != ground_truth.shape:
    raise ValueError("change_map 和 ground_truth 图像大小不一致！")

# 二值化图像（可以根据需要调整阈值）
change_map = (change_map > 128).astype(np.uint8)  # 将灰度值 > 128 的像素视为1（变化）
ground_truth = (ground_truth > 128).astype(np.uint8)  # 将灰度值 > 128 的像素视为1（变化）

# 初始化混淆矩阵图像 (RGB 彩色)
confusion_map = np.zeros((*change_map.shape, 3), dtype=np.uint8)

# 定义 TP, TN, FP, FN 的颜色
white = [255, 255, 255]  # TP 白色
black = [0, 0, 0]  # TN 黑色
green = [0, 255, 0]  # FP 绿色
red = [255, 0, 0]  # FN 红色

# 生成混淆矩阵
TP = (change_map == 1) & (ground_truth == 1)  # True Positive
TN = (change_map == 0) & (ground_truth == 0)  # True Negative
FP = (change_map == 1) & (ground_truth == 0)  # False Positive
FN = (change_map == 0) & (ground_truth == 1)  # False Negative

confusion_map[TP] = white  # TP
confusion_map[TN] = black  # TN
confusion_map[FP] = green  # FP
confusion_map[FN] = red  # FN

# 保存并显示混淆矩阵图像
confusion_image = Image.fromarray(confusion_map)
confusion_image.save('INLPG_california_confusion_map.png')  # 保存为 PNG 图像

# 统计 TP、TN、FP、FN 的数量
TP_count = np.sum(TP)
TN_count = np.sum(TN)
FP_count = np.sum(FP)
FN_count = np.sum(FN)
total_pixels = TP_count + TN_count + FP_count + FN_count

# 计算 OA（Overall Accuracy）
OA = (TP_count + TN_count) / total_pixels

# 计算 P_e（随机一致性概率）
P_yes = ((TP_count + FP_count) / total_pixels) * ((TP_count + FN_count) / total_pixels)
P_no = ((TN_count + FN_count) / total_pixels) * ((TN_count + FP_count) / total_pixels)
P_e = P_yes + P_no

# 计算 Kappa
if 1 - P_e != 0:  # 防止分母为零
    Kappa = (OA - P_e) / (1 - P_e)
else:
    Kappa = 0
# 打印结果
print(f"FP: {FP_count}")
print(f"FN: {FN_count}")
print(f"kappa: {Kappa:.4f}")
print(f"oa: {OA:.4f}")
# 计算 AUC 值
# 将 change_map 和 ground_truth 展开为一维数组
change_map_flat = change_map.flatten()
ground_truth_flat = ground_truth.flatten()

# AUC 计算
auc_value = roc_auc_score(ground_truth_flat, change_map_flat)
print(f"Area Under Curve (AUC): {auc_value:.4f}")
