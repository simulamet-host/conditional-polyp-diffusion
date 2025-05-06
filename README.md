# Installation
To install simply run 
```
conda env create -f environment.yml
```

# Training and Sampling
Training information and additional README can be found in each model subfolder.

## 1. Diffusion model for generating mask(s)
To change directory:
```bash
export OPENAI_LOGDIR={OUTPUT_FOLDER}
```
Model parameters:
```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2"
```
Training:
```bash
python improved-diffusion/scripts/image_train.py --data_dir ./improved-diffusion/datasets/Kvasir-SEG/masks $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
Sampling:
```bash
python improved-diffusion/scripts/image_sample.py --num_samples {SAMPLES} --model_path {MODEL_CHECKPOINT.pt} --output {OUTPUT_TYPE} --postprocess {POSTPROCESS} $MODEL_FLAGS $DIFFUSION_FLAGS
```

## 2. Conditional diffusion model for generating polyp images
Training:
```bash
python latent-diffusion/main.py --base latent-diffusion/configs/latent-diffusion/kvasir-ldm-vq4-.yaml -t --gpus 0,
```
Sampling:
We can generate polyp images based on the condition, based on:
- Existing dataset
```bash
python latent-diffusion/scripts/inference_dataset.py
```
- Generated mask, mask(s) needs to be placed inside ```latent-diffusion/data/samples/masks```:
```bash
python latent-diffusion/scripts/inference_mask.py {IMAGE_NAME} --samples {SAMPLES}
```
Results are stored inside ```latent-diffusion/results```

## Pipeline
To render multiple polyps we can use ```{SAMPLES}``` to sample multiple mask(s) and use them to generate polyp(s):
```bash
export OPENAI_LOGDIR='latent-diffusion/results/masks/'

MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2"

python improved-diffusion/scripts/image_sample.py --num_samples {SAMPLES} --model_path {MODEL_CHECKPOINT.pt} --output png --postprocess {POSTPROCESS} $MODEL_FLAGS $DIFFUSION_FLAGS
python latent-diffusion/scripts/inference_pipe.py
```

## Utils
### a) Mask comparator
To visually inspect overlap between generated mask(s) and the training dataset that was used for training the diffusion model,
we can use:
```bash
python improved-diffusion/scripts/image_compare.py {KVASIR_PATH} {MASK_IMAGE_PATH}
```

### b) Segmentation model
Follow the instruction in [./segmentation_experiments](./segmentation_experiments)

## Contact details:
vajira@simula.no or ro.machacek0@gmail.com


## Citation:
@inproceedings{10.1145/3592571.3592978,
author = {Mach\'{a}\v{c}ek, Roman and Mozaffari, Leila and Sepasdar, Zahra and Parasa, Sravanthi and Halvorsen, P\r{a}l and Riegler, Michael A. and Thambawita, Vajira},
title = {Mask-conditioned latent diffusion for generating gastrointestinal polyp images},
year = {2023},
isbn = {9798400701863},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3592571.3592978},
doi = {10.1145/3592571.3592978},
abstract = {In order to take advantage of artificial intelligence (AI) solutions in endoscopy diagnostics, we must overcome the issue of limited annotations. These limitations are caused by the high privacy concerns in the medical field and the requirement of getting aid from experts for the time-consuming and costly medical data annotation process. In computer vision, image synthesis has made a significant contribution in recent years, as a result of the progress of generative adversarial networks (GANs) and diffusion probabilistic models (DPMs). Novel DPMs have outperformed GANs in text, image, and video generation tasks. Therefore, this study proposes a conditional DPM framework to generate synthetic gastrointestinal (GI) polyp images conditioned on given generated segmentation masks. Our experimental results show that our system can generate an unlimited number of high-fidelity synthetic polyp images with the corresponding ground truth masks of polyps. To test the usefulness of the generated data we trained binary image segmentation models to study the effect of using synthetic data. Results show that the best micro-imagewise intersection over union (IOU) of 0.7751 was achieved from DeepLabv3+ when the training data consists of both real data and synthetic data. However, the results reflect that achieving good segmentation performance with synthetic data heavily depends on model architectures.},
booktitle = {Proceedings of the 4th ACM Workshop on Intelligent Cross-Data Analysis and Retrieval},
pages = {1–9},
numpages = {9},
keywords = {polyp segmentation, polyp generative model, generating synthetic data, diffusion model},
location = {Thessaloniki, Greece},
series = {ICDAR '23}
}

