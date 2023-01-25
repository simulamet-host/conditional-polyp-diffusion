# Installation
To install simply run 
```
conda env create -f environment.yml
```

# Training and Sampling
Training information and additional README can be found in each model subfolder.

## 1. Diffusion model for generating mask(s)
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
python scripts/image_sample.py --model_path {MODEL_CHECKPOINT.pt} --output {OUTPUT_TYPE} --postprocess {POSTPROCESS} $MODEL_FLAGS $DIFFUSION_FLAGS
```
To change outputs do:
```bash
export OPENAI_LOGDIR={OUTPUT_FOLDER}
```

## 2. Conditional diffusion model for generating polyp images
