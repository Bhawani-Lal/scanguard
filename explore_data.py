import os

dataset_path = r"C:\Users\bhawa\.cache\kagglehub\datasets\ravirajsinh45\real-life-industrial-dataset-of-casting-product\versions\2"

for root, dirs,files in os.walk(dataset_path):
    print(f"`Folder:{root}")
    print(" Subfolders:{dirs}")
    print(f" Number of files: {len(files)}")
    if files:
        print(f" Example file: {files[0]}")

    print()