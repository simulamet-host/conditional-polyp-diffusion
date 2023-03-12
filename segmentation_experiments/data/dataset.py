from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            img_dir,
            mask_dir,
            img_exts = ["jpg", "png"],
            mask_exts = ["jpg", "png"],
            classes=[0,255], 
            augmentation=None, 
            preprocessing=None,
            mask_clean=True
    ):
       
        self.img_dir = img_dir
        self.mask_dir  = mask_dir
        
        self.img_exts = img_exts
        self.mask_exts = mask_exts
        
        # Get all image and mask paths using given extensions
        self.images = []
        self.masks = []
        
        for img_ext in self.img_exts:
            self.images.extend(glob.glob(img_dir + f"/*.{img_ext}")) 
            
        for mask_ext in self.mask_exts:
            self.masks.extend(glob.glob(mask_dir + f"/*.{mask_ext}")) 
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.mask_clean = mask_clean
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        
        # Getting images and masks without any mismatch
        img_path = self.images[i]
        img_name = img_path.split("/")[-1] #.split(".")[0]
        mask_path = os.path.join(self.mask_dir, img_name)
        img = Image.open(img_path)
        image_t = np.array(img) # to print, but without preprocessing
        image = np.array(img)
        
        mask = np.array(Image.open(mask_path))[:,:, 0] # take only one channel from 3-channels mask image
        
        #print(mask.min())
        #print(mask.max())
        # Mask clearning (to remove some artifacts coming from labeling)
        if self.mask_clean:
            mask = (mask > 128) # * 255 # if mask value > 128, set it to 255 (this will remove 254, 253 values and 1,2 etc)
        mask = mask.astype("float")
        mask = np.expand_dims(mask, axis=2)
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
       # mask = np.stack(masks, axis=-1).astype('float')

        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            ori_img_smpl = self.augmentation(image=image_t, mask=mask) # here mask is dummy. not required. but need to pass to work
            image, mask = sample['image'], sample['mask']
            image_t = ori_img_smpl["image"]
            

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
  
            
        return {"image":image, "mask":mask, "image_origin": image_t}
        
    def __len__(self):
        return len(self.images)
    
if __name__=="__main__":
    test_dataset = Dataset(img_dir="/work/vajira/DL/roman_diffusion_model/Models/results_135/samples",
                              mask_dir = "/work/vajira/DL/roman_diffusion_model/Models/200")
    print(len(test_dataset))
    #img, mask = 
    sample = test_dataset[500]
    print(sample["image"].shape)
    print(sample["mask"].shape)
    print(sample["image_origin"].shape)
    print(sample["mask"].max())
    print(sample["mask"].min())
    plt.imsave("mask.jpeg", sample["mask"][:,:,0])
    plt.imsave("img.jpeg", sample["image_origin"])