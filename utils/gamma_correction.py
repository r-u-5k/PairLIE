import os

import cv2
import numpy as np


def gamma_correction(img, gamma):
    img_f = img.astype(np.float32) / 255.0

    min_val = img_f.min()
    max_val = img_f.max()
    normalized = (img_f - min_val) / (max_val - min_val)

    gamma_corrected = np.power(normalized, gamma)
    scaled = gamma_corrected * 255.0

    return scaled.astype(np.uint8)

img_folder = 'high'
output_folder = 'mid'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(img_folder):
    path = os.path.join(img_folder, filename)
    img = cv2.imread(path)

    result = gamma_correction(img, gamma=1.8)
    cv2.imwrite(os.path.join(output_folder, filename), result)

print("이미지 처리 완료")
