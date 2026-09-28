import matplotlib.pyplot as plt
from PIL import Image
import os

base_path = r"C:\Users\bhawa\.cache\kagglehub\datasets\ravirajsinh45\real-life-industrial-dataset-of-casting-product\versions\2\casting_data\casting_data\train"

defective_folder = os.path.join(base_path, "def_front")
ok_folder = os.path.join(base_path, "ok_front")

defective_images = os.listdir(defective_folder)[:3]
ok_images = os.listdir(ok_folder)[:3]

fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, img_name in enumerate(defective_images):
    img_path = os.path.join(defective_folder, img_name)
    img = Image.open(img_path)
    axes[0, i].imshow(img)
    axes[0, i].set_title("Defective")
    axes[0, i].axis("off")

for i, img_name in enumerate(ok_images):
    img_path = os.path.join(ok_folder, img_name)
    img = Image.open(img_path)
    axes[1, i].imshow(img)
    axes[1, i].set_title("OK")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("sample_images.png")
print("Saved sample_images.png - open it to view")