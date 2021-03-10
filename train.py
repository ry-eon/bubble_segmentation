import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from valid import bubble_mask_with_segmentation

from utils import Dataset, get_training_augmentation, get_preprocessing, get_preprocessingV
import argparse



def main():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g_num', '--gpu_num', metavar='G', type=str, default='0',
                        help='GPU', dest='gpu_num')
    parser.add_argument('-d', '--device', metavar='G', type=str, default='cpu',
                        help='GPU', dest='device')
    parser.add_argument('-train_dir', '--train_data_dir', type=str, default='/Users/ryeon/Desktop/segmentation_data/',
                        help='dataset dir', dest='train_dir')
    parser.add_argument('-valid_dir', '--valid_data_dir', type=str, default='/Users/ryeon/Desktop/valid_final/result/',
                        help='dataset dir', dest='valid_dir')
    parser.add_argument('-train_epoch', type=int, default=50, help='validation epoch')
    parser.add_argument('-valid_epoch', type=int, default=1, help='train epoch')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained model.pth')
    parser.add_argument('-encoder', type=str, default='mobilenet_v2', help='Encoder')
    parser.add_argument('-encoder_weight', type=str, default='imagenet', help='Encoder')
    parser.add_argument('-activation', type=str, default='sigmoid', help='Encoder')
    parser.add_argument('-simple', type=bool, default=False, help='')
    parser.add_argument('-trans', type=bool, default=False, help='')
    parser.add_argument('-color', type=bool, default=False, help='')
    parser.add_argument('-trans_color', type=bool, default=False, help='')
    args = parser.parse_args()

    print(args)

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weight
    CLASSES = ['bubble']
    ACTIVATION = args.activation
    DEVICE = args.device
    CHANNEL = 3

    model = smp.Unet(
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


    DATA_DIR = args.train_dir
    x_train_dir = os.path.join(DATA_DIR, 'image')
    y_train_dir = os.path.join(DATA_DIR, 'mask')
    VALID_DIR = args.valid_dir
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        train= True,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )


    max_score = 0

    if args.pretrained != None :
        if args.device == 'cuda':
            model = torch.load(args.pretrained)
        else :
            model = torch.load(args.pretrained, map_location = args.device)

    for i in range(0, args.train_epoch):
        print(i, args.train_epoch)

        save_model_name = './train_model_mob_trans_'

        if i % args.valid_epoch == 0:
            print('\nValid: {}'.format(i))
            bubble_mask_with_segmentation(VALID_DIR, model, i,device=args.device)

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)

        # do something (save model, change lr, etc.)
        if max_score < train_logs['iou_score']:
            max_score = train_logs['iou_score']
            os.makedirs('./result/model/', exist_ok=True)
            save_model_name = './result/model/'+save_model_name+str(i)+'.pth'
            torch.save(model, save_model_name)
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        

if __name__ == '__main__':
    main()

