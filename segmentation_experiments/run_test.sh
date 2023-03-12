# !/bin/bash

FNOS="RALLFALL"
model="DeepLabV3plus" # DeepLabV3plus #UnetPlusPlus #FPN
devices="[0]"

for FNO in $FNOS
do
    config_file=config_$FNO.yaml 
    exp_name=$model"_"$config_file 
    out_dir=output/$exp_name"_test_new"
    ckpt_dir=output/$exp_name
     
    echo ======Run Testing -  $exp_name=======
    echo Running Testting - $config_file
    python train.py test --config=$config_file \
                         --trainer.callbacks.init_args.dirpath=$exp_dir \
                         --wandb_name=$exp_name \
                         --ckpt_path=$ckpt_dir/best.ckpt \
                         --output_dir=$out_dir \
                         --model.arch=$model \
                         --trainer.devices $devices \
                         --model.test_print_num 20 \
                         --data.bs 20
    
done
  