import torch
import numpy as np
import argparse
import glob
import os

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from PIL import Image

from ldm.data.kvasir import SegmentationBase


class MaskSeg(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation='bicubic'):
        super().__init__(data_csv='latent-diffusion/data/samples/masks.txt',
                         data_root='latent-diffusion/data/samples/masks',
                         segmentation_root='latent-diffusion/data/samples/masks',
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         n_labels=2)


def ldm_cond_sample_pipe(dataset, config_path, ckpt_path):
    batch_size = 1
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for index, x in enumerate(dataloader):
        seg = x['segmentation']
        name = x['relative_file_path_'][0].split('.')[0]

        with torch.no_grad():
            seg = rearrange(seg, 'b h w c -> b c h w')
            condition = model.to_rgb(seg)

            seg = seg.to('cuda').float()
            seg = model.get_learned_conditioning(seg)

            samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                          ddim_steps=200, eta=1.)
            samples = model.decode_first_stage(samples)

        save_image((samples+1.0)/2.0, f'latent-diffusion/results/{name}.png')
        save_image(condition, f'latent-diffusion/results/{name}_cond.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='latent-diffusion/models/ldm/semantic_synthesis256/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default='latent-diffusion/logs/2022-11-08T20-14-34_kvasir-ldm-vq4-/checkpoints/epoch=000135.ckpt')
    args = parser.parse_args()

    # Generate temporary CSV
    files = glob.glob('latent-diffusion/data/samples/masks/*.png')

    def write_lines(file, lines):
        with open(file, 'w') as f:
            for line in lines:
                f.write(os.path.basename(line))
                f.write('\n')

    write_lines('latent-diffusion/data/samples/masks.txt', files)

    dataset = MaskSeg(size=256)
    ldm_cond_sample_pipe(dataset, args.config_path, args.ckpt_path)
