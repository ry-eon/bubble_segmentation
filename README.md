# Bubble Segmentation
~~~
Note : It's not the final version code. I will the refine and update the code.
~~~

## Overview 
The model segmentates speech bubble within the cut. I have referenced and implemented [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) to segment speech bubble.
In the previous task, the speech bubble detection was performed. In this task, after detection, the speech bubble was accurately segmenized through edge detection such as canny edge detection.
If you are curious about the bubble detection task, refer to the following [bubble detector](https://github.com/Little-BigMan/Bubble-Detector-YOLOv4).
However, performance is limited when edge detector is used(ex: transparency, scatter-type, etc). 
Therefore, masks of some speech bubble were created with edge detector and additional data were collected to create segmentation models.

## Bubble Speech Definition

`Standard`
+ Shape : Ellipse(tawon), Thorn(gasi), Sea_urchin(seonggye), Rectangle(sagak), Cloud(gurm)
+ Color : Black/white(bw), Colorful(color), Transparency(tran), Gradation
+ Form : Basic, Double Speech bubble, Multi-External, Scatter-type
+ ![image](https://user-images.githubusercontent.com/61634628/106093155-1700e500-6173-11eb-9a5e-8828c45271c4.png)
 
<br> 
 
## Initial segmentation
 
`Model`
+ Base Network : Unet
+ Encoder : mobilenet_v2
+ Pretrained : imagenet


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

**The comparison of the three ecoders showed similar performance. Therefore, we chose mobilenet_v2 with the fewest parameters.**

`Improvement Points`
+ Models perform poorly when finding undefined shapes of speech bubble.
+ Models perform poorly when looking for high-transparency speech bubble.
+ Models perform poorly when the internal letters of speech bubbles are distorted.

**Therefore, I solve this problem through data augmentation.**
  
<br>
  
  
  


 ## data augmentation 
 
 ![스크린샷 2021-03-05 오후 6 25 50](https://user-images.githubusercontent.com/61634628/110106579-31228880-7ded-11eb-8949-fc8d8cbfadb7.png)
 
 `augmentation` 
 + Copy to Simple Random Location
 + Copy to Transparent Random Location
 + Copy to Color Random Location
 + Copy to Color + Transparent Random Location
 
 <br>
 
 ## Install dependencies

+ **Pytorch Version** 
    + Pytorch 1.7.0 higher

+ **Install Dependencies Code**
    ~~~
    pip install opencv-python albumentations pillow numpy pretrainedmodels scikit-image scipy segmentation-models-pytorch efficientnet-pytorch timm
    ~~~
    or
    ~~~
    pip install -r requirements.txt
    ~~~
<br>

## Pretrained model 

|**Model**|**Link**|
|---------|--------|
|Mobilenet_v2|[Link](https://drive.google.com/file/d/1kClr7Omvb-REM4r-CrLcjItny7-Zay6p/view?usp=sharing)|
|Mobilenet_v2 + Simple Random Location|[Link](https://drive.google.com/file/d/1Zcxd7H427Gkmv4QbiZCge5E68Of0IiU9/view?usp=sharing)|
|Mobilenet_v2 + Transparent Random Location|[Link]|
|Mobilenet_v2 + Color Random Location|[Link]|
|Mobilenet_v2 + Color + Transparent Random Location|[Link]|

<br>

## Train 
+ **1. Download weight** 

+ **2. Train**     
    ~~~
    python train.py -g gpu_id -dir 'data_dir' -pretrained 'pretrained_model.pth'
    ~~~
    or
    ~~~
    Train.sh 
    ~~~

### Demo    
 
+ **1. Download weight**        
+ **2. Demo**
    ~~~
    python demp.py --weightfile pretrained_model.pth -imgfile image_dir 
    ~~~
