import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

folders = {
    'Celeb-real-train-frame': 0,
    'Celeb-real-test-frame': 0,
    'Celeb-synthesis-train-frame': 1,
    'Celeb-synthesis-test-frame': 1
}

df_list = []

def calculate_gradient_mean(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  

    channel_means = {}
    for i, channel_name in enumerate(['b_mean', 'g_mean', 'r_mean']):
        channel_means[channel_name] = np.mean(np.hypot(grad_x[:, :, i], grad_y[:, :, i]))

    return channel_means

def calculate_gradient_mean_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grad_x = cv2.Sobel(img_hsv, cv2.CV_64F, 1, 0, ksize=3)  
    grad_y = cv2.Sobel(img_hsv, cv2.CV_64F, 0, 1, ksize=3)  

    channel_means = {}
    for i, channel_name in enumerate(['h_mean', 's_mean', 'v_mean']):
        channel_means[channel_name] = np.mean(np.hypot(grad_x[:, :, i], grad_y[:, :, i]))

    return channel_means

def calculate_gradient_mean_ycrcb(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    grad_x = cv2.Sobel(img_ycrcb, cv2.CV_64F, 1, 0, ksize=3)  
    grad_y = cv2.Sobel(img_ycrcb, cv2.CV_64F, 0, 1, ksize=3)  

    channel_means = {}
    for i, channel_name in enumerate(['y_mean', 'cr_mean', 'cb_mean']):
        channel_means[channel_name] = np.mean(np.hypot(grad_x[:, :, i], grad_y[:, :, i]))

    return channel_means

def min_max_normalize(features, min_val, max_val):
    if max_val - min_val != 0:
        normalized_features = (features - min_val) / (max_val - min_val)
    else:
        normalized_features = features
    return normalized_features

def calculate_outlier_proportion_separately(normalized_features, threshold_low=0.2, threshold_high=0.8):

    low_outlier_count = np.sum(normalized_features < threshold_low)
    high_outlier_count = np.sum(normalized_features > threshold_high)
    total_count = len(normalized_features)

    low_outlier_proportion = low_outlier_count / total_count if total_count > 0 else 0
    high_outlier_proportion = high_outlier_count / total_count if total_count > 0 else 0

    return low_outlier_proportion, high_outlier_proportion

all_features = {
    'r_mean': [], 'g_mean': [], 'b_mean': [],
    'h_mean': [], 's_mean': [], 'v_mean': [],
    'y_mean': [], 'cr_mean': [], 'cb_mean': []
}

for folder, label in folders.items():
    folder_path = os.path.join('/home/jonaskao/Data/Celeb-DF-v2/', folder)
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, desc=f'Scanning {folder}'):
            if file.endswith('_face0.jpg'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                img_resized = cv2.resize(img, (128, 128))

                blocks = [img_resized[i:i+32, j:j+32] for i in range(0, 128, 32) for j in range(0, 128, 32)]

                for block in blocks:
                    block_features_bgr = calculate_gradient_mean(block)
                    block_features_hsv = calculate_gradient_mean_hsv(block)
                    block_features_ycrcb = calculate_gradient_mean_ycrcb(block)

                    for k, v in block_features_bgr.items():
                        all_features[k].append(v)
                    for k, v in block_features_hsv.items():
                        all_features[k].append(v)
                    for k, v in block_features_ycrcb.items():
                        all_features[k].append(v)

global_min = {k: np.min(v) for k, v in all_features.items()}
global_max = {k: np.max(v) for k, v in all_features.items()}

for folder, label in folders.items():
    folder_path = os.path.join('/home/jonaskao/Data/Celeb-DF-v2/', folder)
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, desc=f'Processing {folder}'):
            if file.endswith('_face0.jpg'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                img_resized = cv2.resize(img, (128, 128))

                blocks = [img_resized[i:i+32, j:j+32] for i in range(0, 128, 32) for j in range(0, 128, 32)]

                block_features_bgr = [calculate_gradient_mean(block) for block in blocks]
                block_features_hsv = [calculate_gradient_mean_hsv(block) for block in blocks]
                block_features_ycrcb = [calculate_gradient_mean_ycrcb(block) for block in blocks]

                normalized_block_features_bgr = {k: min_max_normalize(np.array([f[k] for f in block_features_bgr]), global_min[k], global_max[k]) for k in block_features_bgr[0]}
                normalized_block_features_hsv = {k: min_max_normalize(np.array([f[k] for f in block_features_hsv]), global_min[k], global_max[k]) for k in block_features_hsv[0]}
                normalized_block_features_ycrcb = {k: min_max_normalize(np.array([f[k] for f in block_features_ycrcb]), global_min[k], global_max[k]) for k in block_features_ycrcb[0]}

                avg_features_bgr = {f'{k}_avg': np.mean(v) for k, v in normalized_block_features_bgr.items()}
                var_features_bgr = {f'{k}_var': np.var(v) for k, v in normalized_block_features_bgr.items()}
                avg_features_hsv = {f'{k}_avg': np.mean(v) for k, v in normalized_block_features_hsv.items()}
                var_features_hsv = {f'{k}_var': np.var(v) for k, v in normalized_block_features_hsv.items()}
                avg_features_ycrcb = {f'{k}_avg': np.mean(v) for k, v in normalized_block_features_ycrcb.items()}
                var_features_ycrcb = {f'{k}_var': np.var(v) for k, v in normalized_block_features_ycrcb.items()}

                low_outliers_bgr = {f'{k}_low_outlier': calculate_outlier_proportion_separately(v)[0] for k, v in normalized_block_features_bgr.items()}
                high_outliers_bgr = {f'{k}_high_outlier': calculate_outlier_proportion_separately(v)[1] for k, v in normalized_block_features_bgr.items()}
                low_outliers_hsv = {f'{k}_low_outlier': calculate_outlier_proportion_separately(v)[0] for k, v in normalized_block_features_hsv.items()}
                high_outliers_hsv = {f'{k}_high_outlier': calculate_outlier_proportion_separately(v)[1] for k, v in normalized_block_features_hsv.items()}
                low_outliers_ycrcb = {f'{k}_low_outlier': calculate_outlier_proportion_separately(v)[0] for k, v in normalized_block_features_ycrcb.items()}
                high_outliers_ycrcb = {f'{k}_high_outlier': calculate_outlier_proportion_separately(v)[1] for k, v in normalized_block_features_ycrcb.items()}

                all_features = {**avg_features_bgr, **var_features_bgr, **low_outliers_bgr, **high_outliers_bgr,
                                **avg_features_hsv, **var_features_hsv, **low_outliers_hsv, **high_outliers_hsv,
                                **avg_features_ycrcb, **var_features_ycrcb, **low_outliers_ycrcb, **high_outliers_ycrcb}

                df_list.append({
                    'folder': folder,
                    'filename': file,
                    'label': label,
                    **all_features
                })

df = pd.DataFrame(df_list)
df.to_csv('/home/jonaskao/Data/Celeb-DF-v2/test.csv', index=False)
print("Finished !")
