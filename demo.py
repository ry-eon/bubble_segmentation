import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import cv2
import os
import copy

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def bubble_mask_with_segmentation(data_dir,segmentation_model, eval_num,device = "cpu") :

    #segmentation_model = torch.load('./demo_best_efficient_b0_backbone_model.pth',map_location='cpu')# this is my previous model
    #segmentation_model = torch.load('./your_model.pth').to(device)  # your model should be here
    #segmentation_model = torch.load('./best_photo1_model.pth',map_location='cpu')  # this is my previous model

    images_dir = os.listdir(data_dir)
    pattern_path ='./pattern.png'
    pattern = cv2.imread(pattern_path)
    pattern = cv2.resize(pattern,(512,512))
    #pattern = np.zeros((512, 512, 3), dtype=np.uint8)
    """
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if i % 2 == 0:
                if j%2 == 0:
                    pattern[i][j] = [0,0,255]
                else :
                    pattern[i][j] = [255,0,0]
            else:
                if j % 2 == 0:
                    pattern[i][j] = [255, 0, 0]
                else:
                    pattern[i][j] = [0, 0, 255]
    """
    """
    step = 8
    w_check = 1
    h_check = 1
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if h_check == 1:
                if w_check == 1 :
                    pattern[i][j] = [0, 0, 255]
                else :
                    pattern[i][j] = [255, 0, 0]
            else:
                if w_check == 1:
                    pattern[i][j] = [255, 0, 0]
                else:
                    pattern[i][j] = [0, 0, 255]
            if j+1 % step == 0:
                 if w_check == 1 :
                     w_check =0
                 else :
                     w_check =1
        if i + 1 % step == 0:
            if h_check == 1:
                h_check = 0
            else:
                h_check = 1
    """



    #print("pattern :",pattern.shape)
    h_stacks = []
    v_stacks = []
    eval_blend = []
    eval_term = 10
    null_img = np.ones((512, 512, 3), dtype=np.uint8)
    #print("start")
    #print("null",null_img.shape)
    for index, name in enumerate(images_dir):
        image_path = os.path.join(data_dir,name )
        pil_image = Image.open(image_path).convert('RGB')
        #pil_image.save("debug.png")
        #print(np.array(pil_image).shape)
        #width, height = pil_image.size
        #print(name)
        width = 512
        height = 512
        pil_image = pil_image.resize((width,height))
        #pil_image = cv2.imread(image_path)
        #print(len(pil_image[0]),len(pil_image))
        #width = len(pil_image[0])
        #height = len(pil_image)
        mask = np.zeros((width, height))

        tensor_image = transform(pil_image)
        tensor_image = tensor_image.to(device).unsqueeze(0)

        #with torch.no_grad():
        pr_mask = segmentation_model.predict(tensor_image)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_mask_pil = Image.fromarray(np.uint8(pr_mask))
        pr_mask_pil = pr_mask_pil.resize((width, height), Image.NEAREST)

        segmentation_mask = np.array(pr_mask_pil)
        segmentation_mask = segmentation_mask.reshape((height, width, 1)) *255
        segmentation_mask = segmentation_mask.astype(np.uint8)
        #print(segmentation_mask.shape)
        mask = copy.deepcopy(segmentation_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #mask = pattern
        segmentation_mask = segmentation_mask.squeeze(axis=2)
        #print(mask.shape)
        #print(segmentation_mask.shape)
        #mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2)
            #print(mask.shape)
            #rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2)
        #rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            #print(rgb_mask.shape)
        #print(mask[segmentation_mask == 255])
        # checkboard image
        # checkboard_mask[segmentation_mask == 255] = checkboard[segmentation_mask == 255]
        mask[segmentation_mask == 255] = pattern[segmentation_mask == 255]

        #print("seg",segmentation_mask.shape,pil_image.shape)
        pil_image = np.array(pil_image)
        pil_image = cv2.cvtColor(pil_image,  cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(pil_image, 0.3, mask, 0.7, 0)
            #print(blend.shape)
        #cv2.imwrite('./evals1.png', mask)
        #cv2.imwrite('./evals2.png', mask)
        #cv2.imwrite('./evals3.png', blend)

        #assert False
        eval_blend.append(blend)
        #assert False
            #pil_image.close()

    #null_img = np.ones((512, 512, 3), dtype=np.uint8) * 255

    for blend in eval_blend:
        #print("check ",blend.shape)
        h_stacks.append(blend)
        if len(h_stacks) == (eval_term ):
            h_stacks = np.hstack(h_stacks)
            #print("in if",np.array(h_stacks).size)
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
        #v_stacks.append(h_stacks)
        #print("shape", np.array(v_stacks).size, np.array(v_stacks[0]).size,np.array(v_stacks[0][0]).size )
        #print("shape", np.array(v_stacks).size)
        v_stacks = np.vstack(v_stacks).astype(np.uint8)

    cv2.imwrite('./evals' +str(eval_num)+'.png', v_stacks)
