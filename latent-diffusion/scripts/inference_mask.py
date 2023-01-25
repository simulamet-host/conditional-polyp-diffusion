import torch
import numpy as np
import argparse

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


def ldm_cond_sample_mask(name, dataset, config_path, ckpt_path, samples):
    batch_size = 1
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x = next(iter(dataloader))
    seg = x['segmentation']

    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        for sample in range(samples):
            samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                          ddim_steps=200, eta=1.)
            samples = model.decode_first_stage(samples)

            save_image((samples+1.0)/2.0, f'latent-diffusion/results/{name}_fake_{sample}.png')
        save_image(condition, f'latent-diffusion/results/{name}_cond.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask', type=str)
    parser.add_argument('--config_path', type=str, default='latent-diffusion/models/ldm/semantic_synthesis256/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default='latent-diffusion/logs/2022-11-08T20-14-34_kvasir-ldm-vq4-/checkpoints/epoch=000135.ckpt')
    parser.add_argument('--samples', type=int, default=1)
    args = parser.parse_args()

    # Generate temporary CSV
    csv_path = 'data/samples/masks.txt'
    f = open(csv_path, "w")
    f.write(args.mask)
    f.close()

    dataset = MaskSeg(size=256)
    name = args.mask.split('.')[0]
    ldm_cond_sample_mask(name, dataset, args.config_path, args.ckpt_path, args.samples)
