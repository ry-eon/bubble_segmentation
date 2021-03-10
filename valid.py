import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import cv2
import os
import copy
import time
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def bubble_mask_with_segmentation(data_dir,segmentation_model, eval_num,device = "cuda") :


    images_dir = os.listdir(data_dir)
    images_dir = [x for x in images_dir if x != '.DS_Store']
    pattern_path ='./pattern.png'
    pattern = cv2.imread(pattern_path)
    pattern = cv2.resize(pattern,(512,512))
    h_stacks = []
    v_stacks = []
    eval_blend = []
    eval_term = 5
    null_img = np.ones((512, 512, 3), dtype=np.uint8)
    for index, name in enumerate(images_dir):
        image_path = os.path.join(data_dir,name )
        pil_image = Image.open(image_path).convert('RGB')
        width = 512
        height = 512
        pil_image = pil_image.resize((width,height))
        mask = np.zeros((width, height))

        tensor_image = transform(pil_image)
        tensor_image = tensor_image.to(device).unsqueeze(0)
        if index == 0 :
            start = time.time()
        pr_mask = segmentation_model.predict(tensor_image)

        if index == 24 :
            end = time.time() - start
            print(eval_num,end)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_mask_pil = Image.fromarray(np.uint8(pr_mask))
        pr_mask_pil = pr_mask_pil.resize((width, height), Image.NEAREST)

        segmentation_mask = np.array(pr_mask_pil)
        segmentation_mask = segmentation_mask.reshape((height, width, 1)) *255
        segmentation_mask = segmentation_mask.astype(np.uint8)
        mask = copy.deepcopy(segmentation_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        segmentation_mask = segmentation_mask.squeeze(axis=2)
        mask[segmentation_mask == 255] = pattern[segmentation_mask == 255]
        pil_image = np.array(pil_image)
        pil_image = cv2.cvtColor(pil_image,  cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(pil_image, 0.5, mask, 0.7, 0)
        eval_blend.append(blend)


    for blend in eval_blend:
        h_stacks.append(blend)
        if len(h_stacks) == (eval_term ):
            h_stacks = np.hstack(h_stacks)
            v_stacks.append(h_stacks)
            h_stacks = []

    if len(h_stacks) < (eval_term * 2) and len(h_stacks) != 0:

        fill_num = int((eval_term * 2 - len(h_stacks)) / 2)
        for _ in range(fill_num):
            h_stacks.append(null_img)
            h_stacks.append(eval_term)
        h_stacks = np.hstack(h_stacks)
    if len(v_stacks) == 0:
        v_stacks = h_stacks
    else:

        v_stacks = np.vstack(v_stacks).astype(np.uint8)
    os.makedirs('./result/image/', exist_ok=True)
    cv2.imwrite('./result/image/check_mob_epoch' +str(eval_num)+'.png', v_stacks)
    #cv2.imwrite( eval_num + '.png', v_stacks)
 #bubble_mask_with_segmentation('/Users/ryeon/Desktop/valid_final/valid/',segmentation_model, eval_num,device = "cuda")