# !/bin/bash

FNOS="RALLFALL"
model="DeepLabV3plus" # other options -> "FPN", "UnetPlusPlus"
devices="[1]"

for FNO in $FNOS
do
    config_file=config_$FNO.yaml
    exp_name=$model"_"$config_file
    exp_dir=output/$exp_name
    
    echo ======Running Training - $exp_name=======
    python train.py fit --config=$config_file \
                        --trainer.callbacks.init_args.dirpath=$exp_dir \
                        --wandb_name=$exp_name \
                        --trainer.callbacks.init_args.save_top_k=1 \
                        --model.arch=$model \
                        --trainer.devices $devices
    
    echo ======Running Testing -  $exp_name=======
    echo Running Testting - $config_file
    python train.py test --config=$config_file \
                         --trainer.callbacks.init_args.dirpath=$exp_dir \
                         --wandb_name=$exp_name \
                         --ckpt_path=$exp_dir/best.ckpt \
                         --output_dir=$exp_dir \
                         --model.arch=$model \
                         --trainer.devices $devices
    
    
done
  