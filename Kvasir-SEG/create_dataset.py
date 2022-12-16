import glob
import os
import random
import shutil


def create_dataset(train=0.7, test=0.2, val=0.1):
    random.seed(0)

    def copy_files(files, destination):
        if not os.path.exists(destination):
            os.makedirs(destination)

        for file in files:
            shutil.copy2(file, destination)

    # Load data
    DATA_DIR = './'

    x_dir = os.path.join(DATA_DIR, 'images')
    y_dir = os.path.join(DATA_DIR, 'masks')

    x_files = glob.glob(f"{x_dir}/*.jpg")
    y_files = glob.glob(f"{y_dir}/*.jpg")

    # Shuffle
    m = list(zip(x_files, y_files))
    random.shuffle(m)
    x_files, y_files = zip(*m)

    # Train, test, val split and copy into folders
    N = len(x_files)
    copy_files(x_files[:int(train * N)], os.path.join(DATA_DIR, "train", "images"))
    copy_files(y_files[:int(train * N)], os.path.join(DATA_DIR, "train", "masks"))

    copy_files(x_files[int(train * N):int((train+test) * N)], os.path.join(DATA_DIR, "test", "images"))
    copy_files(y_files[int(train * N):int((train+test) * N)], os.path.join(DATA_DIR, "test", "masks"))

    copy_files(x_files[int((train+test) * N):], os.path.join(DATA_DIR, "val", "images"))
    copy_files(y_files[int((train+test) * N):], os.path.join(DATA_DIR, "val", "masks"))


if __name__ == '__main__':
    create_dataset()
