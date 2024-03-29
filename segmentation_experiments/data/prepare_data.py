
import pandas as pd
import albumentations as albu
from torch.utils.data import DataLoader, Subset, ConcatDataset
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp

from data.dataset import Dataset
import pytorch_lightning as pl

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        albu.Resize(height=256, width=256, always_apply=True),

       # albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
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
        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        albu.Resize(height=256, width=256, always_apply=True),
    ]
    return albu.Compose(test_transform)



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_float(x, **kwargs):
    return x.astype('float32')



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = []
    if preprocessing_fn:
        _transform.append(albu.Lambda(image=preprocessing_fn))
    _transform.append(albu.Lambda(image=to_tensor, mask=to_tensor))

    
    return albu.Compose(_transform)




# Pytorch lightning DataModule

class PolypDataModule(pl.LightningDataModule):
    
    def __init__(self,  encoder = "se_resnext50_32x4d", 
                 encoder_weight = "imagenet", bs=8, num_workers=2, 
                 data = {
                     "train":[{"img_dir":"", "mask_dir": "", "num_samples": ""},],
                     "validation":[{"img_dir":"", "mask_dir": "", "num_samples": ""},]
                 }
                ):
        
        super().__init__()
        self.conf = OmegaConf.create(data) # create an OmegaConf from input data dictionary
        self.encoder = encoder
        self.encoder_weight = encoder_weight
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weight)
        self.bs = bs
        self.num_workers = num_workers
        
        #print(self.conf)
        
        
    def setup(self, stage: str):
        # Train data
        datasets = []
        #print(self.conf.train)
        for d in self.conf.train:
            data_sub = Dataset( d.img_dir, d.mask_dir, 
                               augmentation=get_training_augmentation(), 
                               preprocessing=get_preprocessing(self.preprocessing_fn))
            if d.num_samples >= 0:
                data_sub = Subset(data_sub, [i for i in range(d.num_samples)])
            datasets.append(data_sub)
        self.train_dataset = ConcatDataset(datasets)
        print("train_dataset_size=", len(self.train_dataset))

        # Valiation data
        datasets = []
        #print(self.conf.validation)
        for d in self.conf.validation:
            data_sub = Dataset(d.img_dir,d.mask_dir,
                               augmentation=get_validation_augmentation(), 
                               preprocessing=get_preprocessing(self.preprocessing_fn))
            if d.num_samples >= 0:
                data_sub = Subset(data_sub, [i for i in range(d.num_samples)])
            datasets.append(data_sub)

        self.validation_dataset = ConcatDataset(datasets)
        print("validation dataset size=", len(self.validation_dataset))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True, num_workers= self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.bs, shuffle=False, num_workers= self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.bs, shuffle=False, num_workers= self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.bs, shuffle=False, num_workers= self.num_workers)
    
    
        
