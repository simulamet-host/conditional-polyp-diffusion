# How to train and test segmenation modles?

## Step 1:

Install requiremnts:

```python
pip install -r requirements.txt
```

## Step 2: train models

Set parameters in run_train.sh:

```shell
FNOS="RALLFALL"

# other options for model -> "FPN", "UnetPlusPlus"
model="DeepLabV3plus" 
devices="[1]"
```
Then, run:

```bash
bash run_train.sh
```

## Step 3: test models and generate samples

```shell
FNOS="RALLFALL"

# Other model options: DeepLabV3plus #UnetPlusPlus #FPN
model="DeepLabV3plus" 
devices="[0]"
```

## Description of input parameters
To be updated...!
