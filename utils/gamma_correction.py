import os

import cv2
import numpy as np


def gamma_correction(img, gamma):
    img_f = img.astype(np.float32)
    min_value = img_f.min()
    max_value = img_f.max()
    img_norm = (img_f - min_value) / (max_value - min_value)
    img_gamma = (img_norm ** gamma) * 255.0
    return img_gamma.astype(np.uint8)


img_folder = 'low'
output_folder = 'light'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(img_folder):
    path = os.path.join(img_folder, filename)
    img = cv2.imread(path)

    result = gamma_correction(img, gamma=0.6)
    cv2.imwrite(os.path.join(output_folder, filename), result)

print("이미지 처리 완료")
