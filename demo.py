import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import copy
import time

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g_num', '--gpu_num', metavar='G', type=str, default='0',
                        help='GPU', dest='gpu_num')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='cpu',
                        help='GPU', dest='gpu')
    parser.add_argument('-image_dir', '--image_data_dir', type=str, default='./test_image/test.png',
                        help='image dir', dest='image_dir')
    parser.add_argument('-pretrained', type=str, default='./pretrained/train_model_uent_8.pth', help='pretrained model.pth')
    parser.add_argument('-encoder', type=str, default='mobilenet_v2', help='Encoder')
    parser.add_argument('-encoder_weight', type=str, default='imagenet', help='Encoder')
    parser.add_argument('-activation', type=str, default='sigmoid', help='Encoder')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weight
    CLASSES = ['bubble']
    ACTIVATION = args.activation
    DEVICE = args.gpu
    CHANNEL = 3

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=CHANNEL,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    if args.gpu == 'cpu':
        model = torch.load(args.pretrained ,map_location='cpu')
    elif args.gpu == 'cuda':
        device = torch.device("cuda")
        model = torch.load(args.pretrained)
        model.to(device)
    else:
        print("device error")
        assert False

    pil_image = Image.open(args.image_dir).convert('RGB')
    width , height = pil_image.size

    pattern_path ='./pattern.png'
    pattern = cv2.imread(pattern_path)
    pattern = cv2.resize(pattern,(width,height))
    null_img = np.ones((width,height, 3), dtype=np.uint8)
    pil_image = pil_image.resize((width, height))
    mask = np.zeros((width, height))

    tensor_image = transform(pil_image)
    tensor_image = tensor_image.to(DEVICE).unsqueeze(0)
    start = time.time()
    pr_mask = model.predict(tensor_image)
    end = time.time() - start
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    pr_mask_pil = Image.fromarray(np.uint8(pr_mask))
    pr_mask_pil = pr_mask_pil.resize((width, height), Image.NEAREST)

    segmentation_mask = np.array(pr_mask_pil)
    segmentation_mask = segmentation_mask.reshape((height, width, 1)) * 255
    segmentation_mask = segmentation_mask.astype(np.uint8)
    mask = copy.deepcopy(segmentation_mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    segmentation_mask = segmentation_mask.squeeze(axis=2)
    mask[segmentation_mask == 255] = pattern[segmentation_mask == 255]

    pil_image = np.array(pil_image)
    pil_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
    demo_image = cv2.addWeighted(pil_image, 0.3, mask, 0.7, 0)
    cv2.imwrite("./result/demo.png", demo_image)
    print("time :", end )
if __name__ == '__main__':
    main()
