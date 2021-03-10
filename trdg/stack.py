import numpy as np
import os
from PIL import Image
import PIL
import cv2

def make_mask(data_dir, save_dir,width,height,save_name):

    imgs = os.listdir(data_dir)
    h_stacks = []
    v_stacks = []
    imgs = [x for x in imgs if x != '.DS_Store']
    for index, name in enumerate(imgs) :
        image_path= os.path.join(data_dir,name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image,(1200,30),interpolation = cv2.INTER_CUBIC)
        h_stacks.append(image)
        if (index+1) % 2 == 0:
            h_stacks = np.array(h_stacks)
            h_stacks = np.hstack(h_stacks)
            v_stacks.append((h_stacks))
            h_stacks = []

    v_stacks = np.vstack(v_stacks).astype(np.uint8)

    v_stacks = cv2.resize(v_stacks,(width,height),interpolation = cv2.INTER_AREA)

    cv2.imwrite(save_dir+save_name+".png",v_stacks)

