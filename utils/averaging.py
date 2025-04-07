import cv2
import os

high_folder = 'middle'
low_folder = 'low'
output_folder = 'middle-low'

os.makedirs(output_folder, exist_ok=True)

filenames = os.listdir(high_folder)

for filename in filenames:
    high_path = os.path.join(high_folder, filename)
    low_path = os.path.join(low_folder, filename)

    img_high = cv2.imread(high_path)
    img_low = cv2.imread(low_path)
    avg_img = cv2.addWeighted(img_high, 0.5, img_low, 0.5, 0)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, avg_img)

print("이미지 처리 완료")
