import os
from PIL import Image

input_base_folder = 'PairLIE-training-dataset'

downscale_factor = 0.5

for folder_name in os.listdir(input_base_folder):
    folder_path = os.path.join(input_base_folder, folder_name)

    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)

            original_size = (img.width, img.height)

            downscaled_size = (int(img.width * downscale_factor), int(img.height * downscale_factor))
            downscaled_img = img.resize(downscaled_size, Image.LANCZOS)
            restored_img = downscaled_img.resize(original_size, Image.BICUBIC)

            new_filename = f"{os.path.splitext(filename)[0]}_{downscaled_size[0]}x{downscaled_size[1]}.png"
            restored_img.save(os.path.join(folder_path, new_filename))

print("해상도 변환 완료")
