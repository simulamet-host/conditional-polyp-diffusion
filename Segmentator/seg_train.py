import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import segmentation_models_pytorch as smp
import ssl
import torch

from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import DataLoader
from utils import *
from shared import *


if __name__ == '__main__':
    x_train_files, x_test_files, x_val_files, y_train_files, y_test_files, y_val_files = load_data()
    print(f'Train: {len(x_train_files)}, Test {len(x_test_files)}, Val: {len(x_val_files)}')

    # Lets look at data we have
    dataset = Dataset(x_train_files, y_train_files, classes=['polyp'])
    image, mask = dataset[4]  # get some sample
    # visualize(image=image, cars_mask=mask.squeeze())

    # Visualize resulted augmented images and masks
    augmented_dataset = Dataset(x_train_files, y_train_files, augmentation=get_training_augmentation(),classes=['polyp'])

    # Show random transforms
    # for i in range(3):
    #     image, mask = augmented_dataset[1]
    #     visualize(image=image, mask=mask.squeeze(-1))

    # To ensure certify is working
    ssl._create_default_https_context = ssl._create_unverified_context

    # Model
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Create datasets
    train_dataset = Dataset(x_train_files, y_train_files, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
    valid_dataset = Dataset(x_val_files, y_val_files, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), classes=CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Losses
    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    # Epoch runners
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Train
    max_score = 0
    for i in range(0, EPOCHS):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # Save model
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
