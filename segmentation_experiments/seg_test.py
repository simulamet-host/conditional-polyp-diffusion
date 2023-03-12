import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models_pytorch as smp
import torch

from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import DataLoader
from utils import *
from shared import *


if __name__ == '__main__':
    # Load data
    x_train_files, x_test_files, x_val_files, y_train_files, y_test_files, y_val_files = load_data()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Load best model
    best_model = torch.load('./best_model.pth')

    # Dataset
    test_dataset = Dataset(x_test_files, y_test_files, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
    test_dataloader = DataLoader(test_dataset)

    # Losses
    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    logs = test_epoch.run(test_dataloader)

    # Test dataset visualization
    test_dataset_vis = Dataset(x_test_files, y_test_files, classes=CLASSES)

    # Visualize
    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )