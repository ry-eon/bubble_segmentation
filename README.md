# Bubble Segmentation

 

## Overview 

The model segmentates speech bubble within the cut. I have referenced and implemented [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) to segment speech bubble.
In the previous task, the speech bubble detection was performed. In this task, after detection, the speech bubble was accurately segmenized through edge detection such as canny edge detection.
If you are curious about the bubble detection task, refer to the following [bubble detector](https://github.com/Little-BigMan/Bubble-Detector-YOLOv4).
However, performance is limited when edge detector is used(ex: transparency, scatter-type, etc). 
Therefore, masks of some speech bubble were created with edge detector and additional data were collected to create segmentation models.
<br>

![result](https://user-images.githubusercontent.com/61634628/110697948-5c173d00-8230-11eb-8865-3433381d3c91.gif)
 
<br> 
 
## Model Configuration and Performance
 
#### Model Configuration

|Model Component|Configuration|
|-----|----|
|Base Network | Unet |
|Encoder | mobilenet_v2|
|Pretrained | imagenet|

#### Model Performance
Inference Images 
   + `size` : 224 x 224
   + `sheets` : 25   

||mobilenet_V2|efficientnet-b0|resnet34|
|----|----|----|---|
|CPU|5.58838 sec|7.83775 sec|9.11186 sec|
|CUDA|||

<!--
+ **CPU**
    + `mobilenet_V2` : 5.588375091552734 sec
    + `resnet34` : 9.111860036849976 sec 
    + `efficientnet-b0` : 7.837748050689697 sec
+ **CUDA**   
    + `mobilenet_V2` : sec
    + `resnet34` : sec 
    + `efficientnet-b0` :  sec
-->
+ The comparison of the three ecoders showed similar performance. Therefore, we chose mobilenet_v2 with the fewest parameters and fastest inference time
    + <details>
        <summary>Compare Encoder</summary>
        <div markdown="1">

        + `Encoder`
            + resnet34
            + efficientnet-b0
            + mobilenet_V2

        + <details>
            <summary>sample 1</summary>
            <div markdown="1">

           |Encoder| Sample|
           |----|----|
           |resnet34|![check_unet_epoch10 png_0](https://user-images.githubusercontent.com/61634628/108817115-bd1cff00-75fa-11eb-8acd-b9d6710394fc.png)|
           |efficientnet-b0|![check_eff_epoch9 png_0](https://user-images.githubusercontent.com/61634628/108944237-b1364900-769d-11eb-9b0e-af07c287b61e.png)|
           |mobilenet_v2|![check_mob_epoch8 png_0](https://user-images.githubusercontent.com/61634628/110071164-e9850800-7dbe-11eb-938a-cbe369a71939.png)|

            </div>
          </details>


          <details>
            <summary>sample 2</summary>
            <div markdown="1">

          |Encoder| Sample|
          |----|----|
          |resnet34|![check_unet_epoch10 png_1](https://user-images.githubusercontent.com/61634628/108817380-2b61c180-75fb-11eb-8040-1851fe383976.png)|
          |efficientnet-b0|![check_eff_epoch9 png_1](https://user-images.githubusercontent.com/61634628/108944337-e0e55100-769d-11eb-8707-36926aeaee82.png)|
          |mobilenet_v2|![check_mob_epoch8 png_1](https://user-images.githubusercontent.com/61634628/110071234-03bee600-7dbf-11eb-81ad-5780829c87c6.png)|

            </div>
          </details> 


          <details>
            <summary>sample 3</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_2](https://user-images.githubusercontent.com/61634628/108817502-53e9bb80-75fb-11eb-8dd6-9fcc3011fbcb.png)|
            |efficientnet-b0|![check_eff_epoch9 png_2](https://user-images.githubusercontent.com/61634628/108944423-0a9e7800-769e-11eb-912e-964ee5c2cc2d.png)|
            |mobilenet_v2|![check_mob_epoch8 png_2](https://user-images.githubusercontent.com/61634628/110071306-20f3b480-7dbf-11eb-82eb-8e2b1f9977e8.png)|

            </div>
          </details> 


          <details>
            <summary>sample 4</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_3](https://user-images.githubusercontent.com/61634628/108817691-9f9c6500-75fb-11eb-9554-5a582b4db04d.png)|
            |efficientnet-b0|![check_eff_epoch9 png_3](https://user-images.githubusercontent.com/61634628/108944518-3c174380-769e-11eb-9a42-509231e980f9.png)|
            |mobilenet_v2|![check_mob_epoch8 png_3](https://user-images.githubusercontent.com/61634628/110071365-3ff24680-7dbf-11eb-9cbd-db27acfb8088.png)|

            </div>
          </details> 


          <details>
            <summary>sample 5</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_4](https://user-images.githubusercontent.com/61634628/108817760-ba6ed980-75fb-11eb-9806-ca265c33061e.png)|
            |efficientnet-b0|![check_eff_epoch9 png_4](https://user-images.githubusercontent.com/61634628/108944647-800a4880-769e-11eb-9877-10486d6f0495.png)|
            |mobilenet_v2|![check_mob_epoch8 png_4](https://user-images.githubusercontent.com/61634628/110071426-58faf780-7dbf-11eb-8fa6-580b2868b9f9.png)|

            </div>
          </details> 


          <details>
            <summary>sample 6</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_5](https://user-images.githubusercontent.com/61634628/108817857-db372f00-75fb-11eb-87ab-f0b86f331545.png)|
            |efficientnet-b0|![check_eff_epoch9 png_5](https://user-images.githubusercontent.com/61634628/108949872-90bfbc00-76a8-11eb-9f95-cc221468672e.png)|
            |mobilenet_v2|![check_mob_epoch8 png_5](https://user-images.githubusercontent.com/61634628/110071469-6e702180-7dbf-11eb-98d0-189054ae3962.png)|

            </div>
          </details> 


          <details>
            <summary>sample 7</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_6](https://user-images.githubusercontent.com/61634628/108817931-f4d87680-75fb-11eb-8c3a-65743a59330e.png)|
            |efficientnet-b0|![check_eff_epoch9 png_6](https://user-images.githubusercontent.com/61634628/108950061-ec8a4500-76a8-11eb-98c8-69c2a53a10cc.png)|
            |mobilenet_v2|![check_mob_epoch8 png_6](https://user-images.githubusercontent.com/61634628/110071517-8778d280-7dbf-11eb-84d7-28e11f92036c.png)|

            </div>
          </details> 


          <details>
            <summary>sample 8</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_7](https://user-images.githubusercontent.com/61634628/108818061-30734080-75fc-11eb-9fc2-0428050d3675.png)|
            |efficientnet-b0|![check_eff_epoch9 png_7](https://user-images.githubusercontent.com/61634628/108950126-0b88d700-76a9-11eb-9527-127bc39684a1.png)|
            |mobilenet_v2|![check_mob_epoch8 png_7](https://user-images.githubusercontent.com/61634628/110071570-9e1f2980-7dbf-11eb-8895-ee69b3dd113f.png)|

            </div>
          </details> 


          <details>
            <summary>sample 9</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_8](https://user-images.githubusercontent.com/61634628/108818197-6284a280-75fc-11eb-9d71-213daadb7ba4.png)|
            |efficientnet-b0|![check_eff_epoch9 png_8](https://user-images.githubusercontent.com/61634628/108950179-23605b00-76a9-11eb-90ca-eb114a6eb82a.png)|
            |mobilenet_v2|![check_mob_epoch8 png_8](https://user-images.githubusercontent.com/61634628/110071640-b98a3480-7dbf-11eb-8259-43b36f9719ec.png)|

            </div>
          </details> 


          <details>
            <summary>sample 10</summary>
            <div markdown="1">

            |Encoder| Sample|
            |----|----|
            |resnet34|![check_unet_epoch10 png_9](https://user-images.githubusercontent.com/61634628/108818291-7fb97100-75fc-11eb-9275-72478ed04fc7.png)|
            |efficientnet-b0|![check_eff_epoch9 png_9](https://user-images.githubusercontent.com/61634628/108950256-40952980-76a9-11eb-9fc3-54e5ff8717f9.png)|
            |mobilenet_v2|![check_mob_epoch8 png_9](https://user-images.githubusercontent.com/61634628/110071689-d161b880-7dbf-11eb-803a-90de609823ba.png)|

            </div>
          </details> 


        </div>
      </details>

#### Pretrained 
|**Model**|**Link**|
|---------|--------|
|Mobilenet_v2|[Link](https://drive.google.com/file/d/1kClr7Omvb-REM4r-CrLcjItny7-Zay6p/view?usp=sharing)|
|Mobilenet_v2 + Simple Random Location|[Link](https://drive.google.com/file/d/1Zcxd7H427Gkmv4QbiZCge5E68Of0IiU9/view?usp=sharing)|
|Mobilenet_v2 + Transparent Random Location|[Link](https://drive.google.com/file/d/1lxiOwGRJHKMnjMagbRDX3JPGX8nNm64M/view?usp=sharing)|
|Mobilenet_v2 + Color Random Location|[Link]|
|Mobilenet_v2 + Color + Transparent Random Location|[Link]|

#### Improvement Points
+ Models perform poorly when finding undefined shapes of speech bubble.
+ Models perform poorly when looking for high-transparency speech bubble.
+ Models perform poorly when the internal letters of speech bubbles are distorted.

**Therefore, I solve this problem through data augmentation.**
  
<br>
  
  
 ## Data Generation
 
 #### overview
 <br>Use [trdg](https://github.com/Belval/TextRecognitionDataGenerator) to generate text data. In the case of trdg, when generating Korean image data, the data is generated one letter at a time. If you want to create a text image consisting of five Korean words, a total of five characters will be created, one by one. Therefore, I added the word in the Korean word dictionary as a txt file. Also, there is only one Korean font in trdg. Therefore, I added a font for Korean. The font is specified as random.

![스크린샷 2021-03-11 오전 2 08 06](https://user-images.githubusercontent.com/61634628/110668469-b0112a00-820e-11eb-88ed-13062326abd7.png)



#### Arguments

+ **Directory**
   + <details>
       <summary>arguments</summary>
       <div markdown="1">
        
        + `--output_dir` : Specify the directory in which to store the generated data.<br>
        + `--input_file` : When set, this argument uses a specified text file as source for the text.
    
       </div> 
    </details>
+ **Text Generation** 
   + <details>
       <summary>arguments</summary>
       <div markdown="1"> 
       
        + `--language` : The language to use, should be fr (French), en (English), es (Spanish), de (German), cn (Chinese), or hi (Hindi).
        + `-c' : The number of images to be created.
        + `-rs` : Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.
        + `-let` : Define if random sequences should contain letters. Only works with -rs
        + `-num` : Define if random sequences should contain numbers. Only works with -rs
        + `-sym` : Define if random sequences should contain symbols. Only works with -rs
        + `-t` : Define the number of thread to use for image generation
        + `-om` : Define if the generator will return masks for the text
       
       </div>
    </details> 
+ **Data Format**
    + <details>
       <summary>arguments</summary>
       <div markdown="1"> 
    
        + `-w` : Define how many words should be included in each generated sample.
        + `-r` : Define if the produced string will have variable word count (with --length being the maximum).
        + `-f` : Define the height of the produced images if horizontal, else the width.
        + `-e` : Define the extension to save the image with.
        + `-wd`: Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used.
        + `-al` : Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right.
        + `-or` : Define the orientation of the text. 0: Horizontal, 1: Vertical.
        + `-sw` : Define the width of the spaces between words. 2.0 means twice the normal space width.
        + `-cs` : Define the width of the spaces between characters. 2 means two pixels.
        + `-m` : Define the margins around the text when rendered. In pixels.
        + `-fi` : Apply a tight crop around the rendered text.
        + `-ca` : Generate upper or lowercase only. arguments: upper or lower. Example: --case upper if you use en.
        + `-ws` : Split on words instead of on characters (preserves ligatures, no character spacing).
        + `-stw` : Define the width of the strokes.
        + `-im` : Define the image mode to be used. RGB is default, L means 8-bit grayscale images, 1 means 1-bit binary images stored with one pixel per byte, etc.
       
       </div>
      </details>
+ **Text Augmentation**
    + <details>
       <summary>arguments</summary>
       <div markdown="1"> 
    
        + `-k` : Define skewing angle of the generated text. In positive degrees.
        + `-rk` : When set, the skew angle will be randomized between the value set with -k and it's opposite.
        + `-bl` : Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius.
        + `-rbl` : When set, the blur radius will be randomized between 0 and -bl.
        + `-b` : Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image.
        + `-na` : Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings.
        + `-d` : Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random.
        + `-do` : Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both.
        + `-tc` : Define the text's color, should be either a single hex color or a range in the ?,? format.
        + `-id` : Define an image directory to use when background is set to image.
        + `-stf` : Define the color of the contour of the strokes, if stroke_width is bigger than.
+ **Mask Generation**
    + <details>
       <summary>arguments</summary>
       <div markdown="1"> 
    
        + `-save_dir` : Specify the directory in which to store the mask image.
        + `-sn` : Define how the produced mask will be named.
        + `-mt` : Defines how many images are used in a row.
        + `-mw` : Define the width of the mask image.
        + `-mh` : Define the height of the mask image.
       </div>
      </details>

#### Implement
+ Use the above argument to generate the data you want.
   ~~~
   python ./trdg/run.py -argument 
   ~~~

  
    




## Data Augmentation 
 
#### overvew
Data augmentation consists of four categories: **Copy to Simple Random Location**, **Copy to Transparent Random Location**, **Copy to Color Random Location**,**Copy to Color + Transparent Random Location**. In case of **Copy to Simple Random Location**, copy the generated speech bubble to a random location inside the cut. In case of **Copy to Transparent Random Location**, make the generated speech bubble transparent and copy it to a random location inside the cut. In case of **Copy to Color Random Location**, color the generated speech bubble and copy it to a random location inside the cut. In case of **Copy to Color + Transparent Random Location**, color the generated speech bubble and make it transparent and copy it to a random location inside the cut. 
 <br><br>
 ![스크린샷 2021-03-05 오후 6 25 50](https://user-images.githubusercontent.com/61634628/110106579-31228880-7ded-11eb-8949-fc8d8cbfadb7.png)
 


 
 <br>
 
 ## Install dependencies

+ **Pytorch Version** 
    + Pytorch 1.7.0 higher

+ **Install Dependencies Code**
    ~~~
    pip install torch torchvision albumentations numpy opencv-python pandas Pillow pretrainedmodels scipy segmentation-models-pytorch efficientnet-pytorch timm requests
    ~~~
    or
    ~~~
    pip install -r requirements.txt
    ~~~
<br>


## Train 
+ **1. Download weight** 
+ **2. Train**     
    
    + `Argument`
        + **device option**
            +  ```-g_num``` : gpu number to use cuda
            +  ```-device``` : Whether the device to be used is cpu or cuda
        + **data option**
            + ```-train_dir``` : The parent folder of the image and mask that you use for training
            + ```-valid_dir``` : The folder of the image and mask that you use for Validating
        + **model option**
            + ```-pretraied``` : pretrained model for the entire network
            + ```-encoder``` : Encoder to use for network. Refer to [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) for encoders.
            + ```-encoder_weight``` : pretrained model for encoder 
            + ```-activation``` :  activation function 
        + **augmentation option** 
            + ```-simple``` : Simply attach the speech bubble to a random location inside the cut.
            + ```-trans``` : Attach the transparent speech bubble to a random location inside the cut.
            + ```-color``` : Attach the color speech bubble to a random location inside the cut. 
            + ```-trans_color``` : Attach the color + transparent speech bubble to a random location inside the cut.


    + `Implement`     
        + **Use the above argument to generate the data you want.** 
        ~~~
        python train.py -g gpu_id -dir 'data_dir' -pretrained 'pretrained_model.pth' ... 
        ~~~


## Demo    
 
+ **1. Download weight**        
+ **2. Demo**
    ~~~
    python demp.py --weightfile pretrained_model.pth -imgfile image_dir 
    ~~~

## Result 

|MobileNet_v2|MobileNet_v2 + trans|
|:---:|:---:|
|![train_model_mob7 pth](https://user-images.githubusercontent.com/61634628/110694235-b366de80-822b-11eb-9008-ff6105f70d44.png)|![train_model_mob_trans_4 pth](https://user-images.githubusercontent.com/61634628/110694209-aba73a00-822b-11eb-9a12-f7ae6dc3ffae.png)|



## Reference 
1. qubvel, [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
2. Belval, [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)

 
