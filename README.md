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
```python
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
python improved-diffusion/scripts/image_sample.py --model_path {MODEL_CHECKPOINT.pt} --output {OUTPUT_TYPE} --postprocess {POSTPROCESS} $MODEL_FLAGS $DIFFUSION_FLAGS
```

## 2. Conditional diffusion model for generating polyp images
Training:
```bash
python main.py --base latent-diffusion/configs/latent-diffusion/kvasir-ldm-vq4-.yaml -t --gpus 0,
```
Sampling:
```bash
python scripts/inference.py
```
We can generate polyp images based on the condition, that is generated mask(s):
```bash
TODO
```

## Utils
### a) Mask comparator
To visually inspect overlap between generated mask(s) and the training dataset that was used for training the diffusion model,
we can use:
```bash
python improved-diffusion/scripts/image_compare.py {KVASIR_PATH} {MASK_IMAGE_PATH}
```
### b) Segmentation model
We can also train segmentation model(s) to compare quality of generated images.
One example is to use Kvasir polyp images to train the model to distinguish polyps and use the model to 
predict polyps in generated polyp images.
