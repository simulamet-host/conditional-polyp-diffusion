#=========================================================
# Developer: Vajira Thambawita (PhD)
# References: * https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
#             * https://pytorch-lightning.readthedocs.io/en/stable/
#=========================================================



import argparse
from omegaconf import OmegaConf
from datetime import datetime
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms,datasets, utils
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
#from torchsummary import summary

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
#from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from einops import rearrange

#from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule
import pandas as pd

#from data.dataset import Dataset
from data.prepare_data import PolypDataModule
#from data.prepare_data import prepare_data_from_multiple_dirs as prepare_data
#from data import PolypsDatasetWithGridEncoding
#from data import PolypsDatasetWithGridEncoding_TestData
#import pyra_pytorch as pyra
#from utils import dice_coeff, iou_pytorch, visualize

import segmentation_models_pytorch as smp
import wandb
from pytorch_lightning.loggers import WandbLogger




# Pytorch lightning training
class PolypModel(pl.LightningModule):

    def __init__(self, 
                 arch="UnetPlusPlus", 
                 encoder_name= "resnet34", 
                 in_channels=3, 
                 out_classes=1, 
                 lr=0.0001, 
                 test_print_batch_id=0, 
                 test_print_num=5, 
                 output_dir="",
                 wandb_name="",
                 **kwargs):
        super().__init__()
        
        
        
        # "UnetPlusPlus", "resnet34", in_channels=3, out_classes=2
        self.arch = arch
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.test_print_batch_id = test_print_batch_id
        self.test_print_num = test_print_num
        self.lr = lr
        self.wandb_name = "test"
        self.output_dir = output_dir
        self.wandb_name = wandb_name
        
        
        
        print(self.output_dir)
        
        #exit(0)
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = smp.create_model(
            self.arch, encoder_name=self.encoder_name, in_channels=self.in_channels, classes=self.out_classes, **kwargs)

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1)) # self.std
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)) # self.mean

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        #print(f"h={h}, w={w}")
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        #if stage == "valid":
        #    print(metrics)
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        #print(image.shape)
        image_before = image
        #print(image_before.shape)
        image_origin = batch["image_origin"]
        mask = batch["mask"]
        
        image = (image - self.mean) / self.std
        mask_p = self.model(image)
        
        prob_mask = mask_p.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        loss = self.loss_fn(mask_p, mask)
        #print(mask_p.shape)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        return {"image_origin": image_origin, 
                "image_before":image_before,
                "mask": mask, 
                "prob_mask": prob_mask, 
                "pred_mask": pred_mask,
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,}  

    def test_epoch_end(self, outputs):
        
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        per_image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        
        per_image_accuray = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        
        per_image_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        
        metrics = {
            "per_image_iou": per_image_iou,
            "per_image_f1": per_image_f1,
            "per_image_accuray": per_image_accuray,
            "per_image_precision": per_image_precision,
            "dataset_iou": dataset_iou,
            "dataset_f1": dataset_f1,
            "dataset_accuracy": dataset_accuracy,
            "dataset_precision": dataset_precision 
        }
        
        self.log_dict(metrics, prog_bar=True)
        
        with open(f"{self.output_dir}/metrics.txt", "w") as f:
          
            for key, value in metrics.items():
                f.write(f"{key}\t={value}")
                f.write("\n")

        #print("test config=", self.ckpt_path)
        
        for i in range(self.test_print_num):
            img = outputs[0]["image_origin"][i].cpu().numpy()
            #img_before = rearrange(outputs[0]["image_before"][i], 'c h w -> h w c').cpu().numpy()
            #print("img before max=", img_before.max())
            #print("img before min=", img_before.min())
            mask = outputs[0]["pred_mask"][i][0,:, :].cpu().numpy()
            mask_gt = outputs[0]["mask"][i][0,:, :].cpu().numpy()
            plt.imsave(f"{self.output_dir}/image_{i}.png",img) 
            plt.imsave(f"{self.output_dir}/mask_pred_{i}.png", mask) 
            plt.imsave(f"{self.output_dir}/mask_gt_{i}.png", mask_gt) 
            #plt.imsave(f"{self.output_dir}/image_before_{i}.png", img_before) 
            
        
            
        # logger.log_image(key=f"samples_{i}", images=[img, mask])
        #plt.imsave("test_pred_mask_1.png",outputs[0]["image_origin"][0][1,:, :].cpu().numpy())
        #img = Image.fromarray(outputs[0]["image"][0].permute(1,2,0).cpu().numpy())
        #img = img.save("test_img.png")
            
    
    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        mask = batch["mask"]
        image = (image - self.mean) / self.std
        mask_p = self.model(image)
        prob_mask = mask_p.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        print(mask_p.shape)
        
        return {"image": image, "mask": mask, "prob_mask": prob_mask, "pred_mask": pred_mask}
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



    
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb_name", default="unet_plus_plus_1")
        parser.add_argument("--wandb_entity", default="simulamet_mlc")
        parser.add_argument("--wandb_project", default="diffusion_polyp")
        parser.add_argument("--output_dir", default="output/new_3")
        #parser.set_defaults({"model.output_dir": "output/test/"})
        
        #parser.set_defaults({"model.config"})
        #parser.link_arguments("trainer.callbacks[" + str(0) +"]", "model.output_dir")
        parser.link_arguments("output_dir", "model.output_dir")
        parser.link_arguments("wandb_name", "model.wandb_name")
        
    #def add_default_arguments_to_parser(self, parser):
        
        
        
    def instantiate_classes(self):
        #print(self.config[self.config.subcommand])
        
        # Call to the logger before initiate other clases, because Trainer class init logger if we didn´t do it
        logger = WandbLogger(entity=self.config[self.config.subcommand].wandb_entity, 
                             project=self.config[self.config.subcommand].wandb_project,
                             name=self.config[self.config.subcommand].wandb_name)
        super().instantiate_classes() # call to super class instatiate_classes()
      
        

# Implementing a CLI
def cli_main():
  
    
    cli = MyLightningCLI(PolypModel, PolypDataModule, 
                       save_config_kwargs={"config_filename": "test_config.yaml", 'overwrite':True}
                       )
    


if __name__ == "__main__":

    
    # Setup Wandb using input yaml file
    #conf = OmegaConf.from_cli() # get input arguments
    #conf_yaml = OmegaConf.load(conf["--config"]) # get input yaml passed in command line
    
    # Setup wandb logger using given entity , project and name (change these as you want)
    #logger = WandbLogger(entity=conf_yaml.wandb_entity, project=conf_yaml.wandb_project,
    #                             name=conf_yaml.wandb_name)
    cli_main() # Pytorch lightning command line interface 
    wandb.finish() # Finish Wandb

   

