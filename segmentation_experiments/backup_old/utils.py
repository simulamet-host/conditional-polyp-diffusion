import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import matplotlib.pyplot as plt
import glob
import albumentations as albu
import numpy as np

from torch.utils.data import Dataset as BaseDataset


def load_data():
    DATA_DIR = '../Kvasir-SEG/'

    x_train_files = glob.glob(os.path.join(DATA_DIR, 'train', 'images') + '/*.jpg')
    y_train_files = glob.glob(os.path.join(DATA_DIR, 'train', 'masks') + '/*.jpg')

    x_test_files = glob.glob(os.path.join(DATA_DIR, 'test', 'images') + '/*.jpg')
    y_test_files = glob.glob(os.path.join(DATA_DIR, 'test', 'masks') + '/*.jpg')

    x_val_files = glob.glob(os.path.join(DATA_DIR, 'val', 'images') + '/*.jpg')
    y_val_files = glob.glob(os.path.join(DATA_DIR, 'val', 'masks') + '/*.jpg')

    return x_train_files, x_test_files, x_val_files, y_train_files, y_test_files, y_val_files


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=480, width=480, always_apply=True),
        albu.Resize(height=512, width=512, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 512),
        albu.Resize(height=512, width=512, always_apply=True),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class Dataset(BaseDataset):
    CLASSES = ['polyp']

    def __init__(
            self,
            images_files,
            masks_files,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = [os.path.basename(filepath) for filepath in images_files]
        self.images_fps = images_files
        self.masks_fps = masks_files

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
