import os

dataset_path = "data"

for root, dirs,files in os.walk(dataset_path):
    print(f"`Folder:{root}")
    print(" Subfolders:{dirs}")
    print(f" Number of files: {len(files)}")
    if files:
        print(f" Example file: {files[0]}")

    print()