import os

import cv2
import numpy as np


def gamma_correction(img_high, img_low, alpha=0.5, gamma=2.2):
    """
    sRGB 역감마 → 선형 보간 → 정감마
    img_high, img_low: uint8 HxWx3, [0,255]
    alpha: high 대비 low 비율 (0.0 ~ 1.0)
    """
    # 1) [0,1] float
    h = img_high.astype(np.float32) / 255.0
    l = img_low.astype(np.float32) / 255.0

    # 2) 역감마 (sRGB → 선형)
    h_lin = np.power(h, gamma)
    l_lin = np.power(l, gamma)

    # 3) 선형 보간
    mid_lin = (1 - alpha) * h_lin + alpha * l_lin

    # 4) 정감마 (선형 → sRGB)
    mid = np.power(mid_lin, 1.0 / gamma)

    # 5) [0,255] uint8
    return (np.clip(mid * 255.0, 0, 255)).astype(np.uint8)


# 입력/출력 폴더 설정
high_folder = 'high'
low_folder = 'low'
output_folder = 'middle'

os.makedirs(output_folder, exist_ok=True)

# high 폴더의 모든 파일 처리
for filename in os.listdir(high_folder):
    high_path = os.path.join(high_folder, filename)
    low_path = os.path.join(low_folder, filename)

    # 파일 유효성 검사
    if not os.path.isfile(high_path) or not os.path.isfile(low_path):
        continue

    img_high = cv2.imread(high_path)
    img_low = cv2.imread(low_path)
    if img_high is None or img_low is None:
        continue

    # 감마 보정 후 선형 평균
    mid_img = gamma_correction(img_high, img_low, alpha=0.5, gamma=2.2)

    # 저장
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, mid_img)

print("이미지 처리 완료")
