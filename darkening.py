import os
from PIL import Image, ImageEnhance

dataset_dir = 'PairLIE-training-dataset'
img_extensions = ['.jpg', '.jpeg', '.png']

for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path)
                       if any(f.lower().endswith(ext) for ext in img_extensions)]

        if len(image_files) == 2:
            file_name = image_files[0]
            img_path = os.path.join(folder_path, file_name)
            image = Image.open(img_path)

            enhancer = ImageEnhance.Brightness(image)
            new_image = enhancer.enhance(0.8)

            base_name, ext = os.path.splitext(file_name)
            new_img_filename = base_name + "_new" + ext
            new_img_path = os.path.join(folder_path, new_img_filename)
            new_image.save(new_img_path)
