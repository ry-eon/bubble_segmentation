import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from demo import bubble_mask_with_segmentation

from utils import Dataset, get_training_augmentation, get_preprocessing, get_preprocessingV


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['bubble']
    ACTIVATION = 'sigmoid'
    DEVICE = 'cpu'
    CHANNEL = 3

    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels = CHANNEL,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])


    #DATA_DIR = '/Users/ryeon/Desktop/segmentation_data/'
    DATA_DIR = '/content/gdrive/MyDrive/segmentation/data/'
    x_train_dir = os.path.join(DATA_DIR, 'image')
    y_train_dir = os.path.join(DATA_DIR, 'mask')
    #print(DATA_DIR,x_train_dir,y_train_dir)
    #VALID_DIR = '/Users/ryeon/Desktop/valid/'
    VALID_DIR = '/content/gdrive/MyDrive/segmentation/data/'
    x_valid_dir = os.path.join(DATA_DIR, 'valid_final')
    #x_valid_dir = VALID_DIR
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    valid_dataset = Dataset(
        images_dir = VALID_DIR,
        masks_dir = None,
        preprocessing=get_preprocessingV(preprocessing_fn),
        classes=CLASSES,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )


    max_score = 0
    save_model_name = './train_model'

    for i in range(0, 100):


        #valid_log = valid_epoch.run(valid_loader)
        if i % 3 == 0:
            print('\nValid: {}'.format(i))
            bubble_mask_with_segmentation(x_valid_dir, model, i)

        #assert False
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)

        # do something (save model, change lr, etc.)
        if max_score < train_logs['iou_score']:
            max_score = train_logs['iou_score']
            save_model_name = save_model_name+str(i)+'.pth'
            torch.save(model, save_model_name)
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    main()

