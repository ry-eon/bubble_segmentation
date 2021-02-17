import argparse
from easydict import EasyDict as edict
import os
import numpy as np
import sys
import torch
from PIL import Image
import PIL
import cv2
import torchvision.transforms as transforms


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

class dataset_loader(object):
    def __init__(self, parser):
        self.ROOT = parser.dataset_dir

        self.imgs = os.listdir(os.path.join(self.ROOT, 'image'))
        self.xmls = os.listdir(os.path.join(self.ROOT, 'XML'))
        #print(self.imgs)
        #assert False
        self.imgs_name = [x.split('.')[0] for x in self.imgs]
        self.xmls = [x for x in self.xmls if x.split('.')[-1] == 'xml']
        self.xmls = [x for x in self.xmls if len(self.xml_bbox(os.path.join(self.ROOT, 'XML', x))) != 0]
        # xml 파일 기준으로 매칭되는 이미지가 있는지  검사
        self.xmls = [x for x in self.xmls if x.split('.')[0] in self.imgs_name]
        # image 파일 기준으로 매칭되는 xml파일이 있는 검사
        self.xmls_name = [x.split('.')[0] for x in self.xmls]
        self.imgs = [x for x in self.imgs if x.split('.')[0] in self.xmls_name]

        # image , xml file path sort
        self.imgs.sort()
        self.xmls.sort()
        for i in range(len(self.imgs)):
            print("imgs:{}    xml:{} ".format(self.imgs[i], self.xmls[i]))
        assert len(self.imgs) == len(self.xmls) != 0, 'data  number error!! imgs {} / xmls {}'.format(len(self.imgs),


                                                                                                  len(self.xmls))
    def xml_bbox(self, xml_path):
        res = []
        target = ET.parse(xml_path).getroot()
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin','ymin','xmax','ymax']
            bndbox = []
            for i , pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text)-1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind = 0
            bndbox.append(label_idx)
            res+=[bndbox]
        return np.array(res, dtype = np.float32)


    def bbox_loader(self, index):
        img_path = os.path.join(self.ROOT, 'image', self.imgs[index])
        truth = self.xml_bbox(os.path.join(self.ROOT, 'XML', self.xmls[index]))
        bboxes = np.array(truth)
        print(bboxes)
        return img_path, bboxes




    def bubble_mask_with_segmentation(self):
        segmentation_model = torch.load('demo_best_efficient_b0_backbone_model.pth',map_location='cpu')  # this is my previous model
        device='cpu'
        for index  in range(len(self.xmls)):
            img_path = os.path.join(self.ROOT, 'image', self.imgs[index])
            truth = self.xml_bbox(os.path.join(self.ROOT, 'XML', self.xmls[index]))
            bboxes = np.array(truth)
            pil_image = Image.open(img_path)

            width, height = pil_image.size

            mask = np.zeros((width,height))

            for i in range(len(bboxes)):
                #print(i , bboxes[i][0],bboxes[i][1],bboxes[i][2],bboxes[i][3])

                box = pil_image.crop((bboxes[i][0],bboxes[i][1],bboxes[i][2],bboxes[i][3] ))

                box_width, box_height = box.size
                print("box : " ,box_width, box_height)
                tensor_image = transform(box)
                tensor_image = tensor_image.to(device).unsqueeze(0)

                pr_mask = segmentation_model.predict(tensor_image)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())

                pr_mask_pil = Image.fromarray(np.uint8(pr_mask))
                pr_mask_pil = pr_mask_pil.resize((box_width, box_height), Image.NEAREST)

                segmentation_mask = np.array(pr_mask_pil)
                print("seg :", len(segmentation_mask),len(segmentation_mask[0])  )
                print(len(segmentation_mask), len(segmentation_mask[0]),int(bboxes[i][3]-bboxes[i][1]),int(bboxes[i][2]-bboxes[i][0])  )

                mx = int(bboxes[i][0])
                Mx = int(bboxes[i][2])
                my = int(bboxes[i][1])
                My = int(bboxes[i][3])
                print("ptr" , mx,Mx,my,My)
                mask[my:My,mx:Mx] = segmentation_mask



            mask = mask.reshape((height, width, 1))

            name = str(self.imgs[index]).split(".")[0]
            print(name)
            pil_image.close()
            cv2.imwrite("./"+name+".png", mask * 255)









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This code create mask(segmentation) data using bounding box',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help=' best_efficient_b0_backbone_model.pth')
    parser.add_argument('-classes', type=int, default=1, help='dataset classes')
    parser.add_argument('-size', type=int, default=None, help='dataset image size')
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')

    bbox_dataLoader = dataset_loader(args)
    bbox_dataLoader.bubble_mask_with_segmentation()


