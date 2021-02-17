import argparse
from easydict import EasyDict as edict
import os
import numpy as np
import sys
import torch
from PIL import Image
import PIL
import cv2





def load_image(args):
    print("image root", args.img_dataset_dir)
    print("mask root", args.mask_dataset_dir)
    imgs = os.listdir(args.img_dataset_dir)
    masks = os.listdir(args.mask_dataset_dir)
    imgs_name = [x.split('.')[0] for x in imgs]
    masks = [x for x in masks if x.split('.')[-1] == 'png']

    # xml 파일 기준으로 매칭되는 이미지가 있는지  검사
    masks = [x for x in masks if x.split('.')[0] in imgs_name]
    # image 파일 기준으로 매칭되는 xml파일이 있는 검사
    mask_name = [x.split('.')[0] for x in masks]
    imgs = [x for x in imgs if x.split('.')[0] in mask_name]
    imgs.sort()
    masks.sort()
    for i in range(len(imgs)):
        print("imgs:{}    xml:{} ".format(imgs[i], masks[i]))
    assert len(imgs) == len(masks) != 0

    for index in range(len(imgs)) :
        alpha = 0.7
        img_path = os.path.join(args.img_dataset_dir, imgs[index])
        mask_path = os.path.join(args.mask_dataset_dir, masks[index])
        print("image path",img_path)
        print("mask path", mask_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        rgb_mask = cv2.imread(mask_path)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2)
        print("check", img.shape, mask.shape, rgb_mask.shape)
        rgb_mask[mask == 255] = [0, 0, 255]
        print("check", img.shape, mask.shape, rgb_mask.shape)
        assert False
        print("img :", len(img),len(img[0]))
        print("mask :", len(mask), len(mask[0]))

        blend = cv2.addWeighted(img,0.8,rgb_mask,0.2,0)
        name = mask_path.split(".")[-2].split("/")[-1]

        cv2.imwrite("./trans/"+name+".png",blend)

"""
def load_image(img_dataset_dir,mask_dataset_dir):
    print("image root", img_dataset_dir)
    print("mask root", mask_dataset_dir)
    imgs = os.listdir(img_dataset_dir)
    masks = os.listdir(mask_dataset_dir)
    imgs_name = [x.split('.')[0] for x in imgs]
    masks = [x for x in masks if x.split('.')[-1] == 'png']

    # xml 파일 기준으로 매칭되는 이미지가 있는지  검사
    masks = [x for x in masks if x.split('.')[0] in imgs_name]
    # image 파일 기준으로 매칭되는 xml파일이 있는 검사
    mask_name = [x.split('.')[0] for x in masks]
    imgs = [x for x in imgs if x.split('.')[0] in mask_name]
    imgs.sort()
    masks.sort()
    for i in range(len(imgs)):
        print("imgs:{}    xml:{} ".format(imgs[i], masks[i]))
    assert len(imgs) == len(masks) != 0

    for index in range(len(imgs)) :
        alpha = 0.7
        img_path = os.path.join(img_dataset_dir, imgs[index])
        mask_path = os.path.join(mask_dataset_dir, masks[index])
        print("image path",img_path)
        print("mask path", mask_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        rgb_mask = cv2.imread(mask_path)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2)
        print("check",img.shape,mask.shape,rgb_mask.shape)
        rgb_mask[mask == 255] = [0, 0, 255]
        assert False
        print("img :", len(img),len(img[0]))
        print("mask :", len(mask), len(mask[0]))

        blend = cv2.addWeighted(img,0.8,rgb_mask,0.2,0)
        name = mask_path.split(".")[-2].split("/")[-1]

        cv2.imwrite("./trans_/"+name+".png",blend)
"""















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This code create mask(segmentation) data using bounding box',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-img_dir', '--img-data-dir', type=str, default=None,
                        help='image dataset dir', dest='img_dataset_dir')
    parser.add_argument('-mask_dir', '--mask-data-dir', type=str, default=None,
                        help='mask dataset dir', dest='mask_dataset_dir')
    args = parser.parse_args()
    load_image(args)



