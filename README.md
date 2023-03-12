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