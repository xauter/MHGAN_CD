import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def compute_confusion_metrics(conf_map_path):
    # Load the confusion map image (PNG)
    conf_map = mpimg.imread(conf_map_path)

    # Ensure it's in the range [0, 1]
    conf_map = conf_map.astype(np.float32)

    # Extract the individual channels (Target change map, Change map, and AND result)
    target_change_map = conf_map[:, :, 0]
    change_map = conf_map[:, :, 1]
    and_map = conf_map[:, :, 2]

    # Compute TP, FP, FN, TN
    TP = np.sum(np.logical_and(target_change_map == 1, change_map == 1))
    FP = np.sum(np.logical_and(target_change_map == 0, change_map == 1))
    FN = np.sum(np.logical_and(target_change_map == 1, change_map == 0))
    TN = np.sum(np.logical_and(target_change_map == 0, change_map == 0))

    # Print the results
    print(f"True Positive (TP): {TP}")
    print(f"False Positive (FP): {FP}")
    print(f"False Negative (FN): {FN}")
    print(f"True Negative (TN): {TN}")

    return TP, FP, FN, TN
def main():
    conf_map_path = r"D:\lxy\TSCNet-master\TSCNet\logs\California\20241226-094549\images\z_Confusion_map.png"
    # 计算指标
    compute_confusion_metrics(conf_map_path)



if __name__ == "__main__":
    main()