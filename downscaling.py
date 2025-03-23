import os
from PIL import Image

input_base_folder = 'PairLIE-training-dataset'

scales = [1, 0.5, 0.25]

for folder_name in os.listdir(input_base_folder):
    folder_path = os.path.join(input_base_folder, folder_name)

    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)

            for scale in scales:
                if scale == 1:
                    continue

                new_size = (int(img.width * scale), int(img.height * scale))
                resized_img = img.resize(new_size, Image.LANCZOS)

                new_filename = f"{os.path.splitext(filename)[0]}_downscaled_{new_size[0]}x{new_size[1]}.png"
                resized_img.save(os.path.join(folder_path, new_filename))

print("해상도 변환 완료")
