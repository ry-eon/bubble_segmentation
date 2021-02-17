import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu


class Dataset(BaseDataset):


    CLASSES = ['bubble']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            train = True,
    ):

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
            #print(self.imgs_dir, self.masks_dir)
            #assert False
            self.imgs_name = [x.split('.')[0] for x in self.imgs_dir]



            self.masks_dir = [x for x in self.masks_dir if x.split('.')[0] in self.imgs_name and x.split(".")[-1]=="png" ]
            self.imgs_dir.sort()
            self.masks_dir.sort()

            self.masks_name = [x.split('.')[0] for x in self.masks_dir]
            self.imgs_dir = [x for x in self.imgs_dir if x.split('.')[0] in self.masks_name]


            self.imgs_dir.sort()
            self.masks_dir.sort()

            #assert False
            assert len(self.imgs_dir) == len(self.masks_dir) != 0


#        self.ids = self.make_list()

 #       self.images_fps = [os.path.join(images_dir, mask_id + ".jpg") for mask_id in self.ids]
 #       self.masks_fps = [os.path.join(masks_dir, mask_id + ".png") for mask_id in self.ids]


        #assert False
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    """"
    def make_list(self):

        mask_list = os.listdir(self.masks_dir)
        image_list = os.listdir(self.images_dir)

        name_mask_list = []
        name_intersection_list = []
        
        for mask_file in mask_list:
            file_name = mask_file.split(".")[0]
            name_mask_list.append(file_name)

        for image_file in image_list:
            file_name = image_file.split(".")[0]
            if file_name in name_mask_list:
                name_intersection_list.append(file_name)

        return name_intersection_list
    """
    def __getitem__(self, i):

        # read data
        image = cv2.imread(os.path.join(self.imgs_root,self.imgs_dir[i]))
        #print('img', image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            mask = cv2.imread(os.path.join(self.masks_root ,self.masks_dir[i]), 0)

        # extract certain classes from mask (e.g. cars)
        #print(np.unique(mask))
            masks = [(mask == 255)]
            mask = np.stack(masks, axis=-1).astype('float')
        #print(image.shape, mask.shape)
        # apply augmentations
        if self.augmentation:
            #print("aug")
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']


        # apply preprocessing
        if self.preprocessing:
            #print("pre")
            if self.train:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            #print("end pre")
            else :
                sample = self.preprocessing(image=image)
                image = sample['image']

        if self.train:
            return image, mask
        else :
            return image
    def __len__(self):
        return len(self.imgs_dir)


def get_training_augmentation():

    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Resize(224, 224)
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    #print(x.shape)
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