import numpy as np
import argparse
from PIL import Image
from scipy.stats import ks_2samp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, help='Global path to image file (npz)')
    args = parser.parse_args()

    IMGS = np.load(args.images)
    imgs = IMGS['arr_0']
    n, h, w, ch = imgs.shape
    for i in range(n):
        img = Image.fromarray(imgs[i, :, :, :])
        img = img.resize((128, 128))
        print(img)
        img.show()
