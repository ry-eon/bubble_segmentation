import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import random


class Dataset(BaseDataset):
    CLASSES = ['bubble']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            train=True,

    ):
        self.check = 0
        self.imgs = cv2.imread('300.png', cv2.IMREAD_UNCHANGED)
        self.letter = cv2.imread('letter.png', cv2.IMREAD_UNCHANGED)
        self.letter = cv2.cvtColor(self.letter, cv2.COLOR_BGR2BGRA)
        alpha_map = self.imgs[:, :, 3].copy()
        self.black = np.zeros((self.imgs.shape[0], self.imgs.shape[1], 1), dtype=np.uint8)
        self.black[alpha_map > 127] = 255
        self.nLabels, self.labels, self.stats, _ = cv2.connectedComponentsWithStats(self.black, connectivity=8)
        b, g, r, a = cv2.split(self.imgs)
        bool_map = (b > 127) & (g > 127) & (r > 127) & (a > 127)
        self.imgs[bool_map] = self.letter[bool_map]
        self.train = train
        if self.train == False:
            self.imgs_root = images_dir
            self.imgs_dir = os.listdir(images_dir)
            self.train = False



        else:
            self.imgs_root = images_dir
            self.masks_root = masks_dir

            self.imgs_dir = os.listdir(images_dir)
            self.masks_dir = os.listdir(masks_dir)
            self.imgs_dir.sort()
            self.masks_dir.sort()

            self.imgs_name = [x.split('.')[0] for x in self.imgs_dir]

            self.masks_dir = [x for x in self.masks_dir if
                              x.split('.')[0] in self.imgs_name and x.split(".")[-1] == "png"]
            self.imgs_dir.sort()
            self.masks_dir.sort()

            self.masks_name = [x.split('.')[0] for x in self.masks_dir]
            self.imgs_dir = [x for x in self.imgs_dir if x.split('.')[0] in self.masks_name]

            self.imgs_dir.sort()
            self.masks_dir.sort()

            assert len(self.imgs_dir) == len(self.masks_dir) != 0

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, i):

        # read data
        image = cv2.imread(os.path.join(self.imgs_root, self.imgs_dir[i]))


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            mask = cv2.imread(os.path.join(self.masks_root, self.masks_dir[i]), 0)
        if self.augmentation:

            if random.randint(0, 10) > 4:
                image, mask = self.bubble_copy(image, mask, Trans=True)
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            masks = [(mask == 255)]
            mask = np.stack(masks, axis=-1).astype('float')
        if self.preprocessing:
            if self.train:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.preprocessing(image=image)
                image = sample['image']

        if self.train:
            return image, mask
        else:
            return image

    def __len__(self):
        return len(self.imgs_dir)

    def bubble_copy(self, image, mask, Trans=False):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        while (True):
            bubble_index = np.random.randint(self.nLabels)
            if self.stats[bubble_index][4] > 200:
                break
        # bubble_index =14
        # print(bubble_index)
        canvas = self.labels.copy()
        canvas[self.labels == (bubble_index)] = 255
        canvas[self.labels != (bubble_index)] = 0
        x, y, w, h, _ = self.stats[bubble_index]
        # cv2.imwrite("canvas.png",canvas)
        temp2 = self.imgs.copy()
        temp2[canvas == 0] = 0
        # cv2.imwrite("imags2.png",self.imgs)
        bubble_image = temp2[y: y + h, x:x + w, :]
        # cv2.imwrite("bubble_image.png",bubble_image)
        # print(self.black.shape)
        temp1 = self.black.copy()
        temp1[canvas == 0] = 0
        temp1[canvas != 0] = 255
        # print(temp.shape)
        bubble_mask = temp1[y: y + h, x:x + w, :]
        # print(bubble_mask.shape, mask.shape)
        bubble_image = cv2.resize(bubble_image, (120, 90), cv2.INTER_AREA)
        bubble_mask = cv2.resize(bubble_mask, (120, 90), cv2.INTER_AREA)
        bubble_mask[bubble_mask > 0] = 255
        # bubble_mask = bubble_mask.reshape((90,120,1))
        r_w = random.randint(0, image.shape[0] - 120)
        r_h = random.randint(0, image.shape[1] - 90)

        if Trans:
            letter_mask = np.ones((90, 120, 4)) * 255
            bubble_ = cv2.cvtColor(bubble_image, cv2.COLOR_BGRA2GRAY)

            letter_mask[bubble_ < 175] = bubble_image[bubble_ < 175]
            letter_mask[:, :, 0][bubble_ < 127] = 0
            letter_mask[:, :, 1][bubble_ < 127] = 0
            letter_mask[:, :, 2][bubble_ < 127] = 0
            letter_mask[:, :, 3][bubble_ >= 175] = 0
            ratio = np.random.randint(5, 9) / 10.0
            temp = image[r_h: r_h + 90, r_w: r_w + 120]
            temp_image = cv2.addWeighted(temp, 1, bubble_image, ratio, 0)
            self.check += 1
            image[r_h: r_h + 90, r_w: r_w + 120] = temp_image
            image[r_h: r_h + 90, r_w: r_w + 120][letter_mask[:, :, 3] != 0] = letter_mask[letter_mask[:, :, 3] != 0]
        else:
            image[r_h: r_h + 90, r_w: r_w + 120][bubble_image[:, :, 3] > 127] = bubble_image[
                bubble_image[:, :, 3] > 127]
        mask[r_h: r_h + 90, r_w: r_w + 120][bubble_image[:, :, 3] > 127] = bubble_mask[bubble_image[:, :, 3] > 127]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return image, mask


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(224, 224)
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    # print(x.shape)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_preprocessingV(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

