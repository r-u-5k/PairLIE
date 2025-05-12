import os

import cv2
import numpy as np


def gamma_correction(img, gamma=0.4):
    img_f = img.astype(np.float32) / 255.0
    corrected = np.power(img_f, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


img_folder = 'high'
output_folder = 'mid'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(img_folder):
    path = os.path.join(img_folder, filename)
    img = cv2.imread(path)

    result = gamma_correction(img, gamma=1.8)
    cv2.imwrite(os.path.join(output_folder, filename), result)

print("이미지 처리 완료")
