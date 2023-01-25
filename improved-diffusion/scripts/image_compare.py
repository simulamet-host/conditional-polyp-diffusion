import numpy as np
import argparse
import os
import cv2
from PIL import Image
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def concat_h(A, B):
    res = Image.new('RGB', (A.width + B.width, A.height))
    res.paste(A, (0, 0))
    res.paste(B, (A.width, 0))
    return res


def similarity(A, B):
    return np.sum(A == B)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, help='Global path to training directory')
    parser.add_argument('image', type=str, help='Global path to image file (png)')
    args = parser.parse_args()

    img_sample = Image.open(args.image).convert('L')
    w, h = img_sample.size
    img_max_size = w if w > h else h
    img_size = (img_max_size, img_max_size)

    img_sample = img_sample.resize(img_size)
    np_sample = np.array(img_sample)

    img_closest = None, None
    sim_closest = -np.inf
    for path_train in os.listdir(args.training):
        path_train = os.path.join(args.training, path_train)
        img_train = Image.open(path_train).convert('L')
        img_train = img_train.resize(img_size)
        np_train = np.array(img_train)

        s = similarity(np_train, np_sample)
        if s > sim_closest:
            img_closest = img_train
            sim_closest = s

    print("similarity: ", 100*sim_closest/(img_max_size*img_max_size), '%')
    concatenated = concat_h(img_sample, img_closest)
    concatenated.show()
