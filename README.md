# Post-Disaster-Semenatic-Segmentation

![download (11)](https://user-images.githubusercontent.com/117116368/218915983-da8ef616-fc88-49b6-b031-90f7faa0aa9e.png)
## Table of Contents

* [Overview](#Overview)
* [Requirements](#Requirements)
* [Dataset](#Dataset)
* [References](#References)
* [Contributions](#Contributions)

## Overview
This repository contains code for a deep learning based semantic segmentation model for scene parsing of high-quality, low-resolution, post-disaster UAV imagery analysis. This project is motivated by the increasing frequency and severity of floods, which can cause widespread damage to infrastructure and disrupt lives. In order to effectively respond to natural disasters, it is important to quickly assess the extent of damage and identify areas that need priority attention. The model is trained to classify 10 different classes including Background, Building-Flooded, Building-NonFlooded,Road, Road-NonFlooded, Tree, Pool, Water, Vehicle, and Grass of disaster stricken areas. 

The process of UAV image classification involves two stages: semantic segmentation and pixel-wise classification. Semantic segmentation breaks down an image into uniform regions, allowing us to understand the local structure of the image and distinguish between boundaries such as lines and curves. This process involves assigning a label to each pixel in the image, where pixels with the same label possess similar visual attributes.

## Requirements
The code in this repository is written in Python and requires the following packages:
* Segmentation-Models
* Tensorflow
* Keras
* Numpy
* OpenCV
* Matplotlib

## Dataset
The original dataset consists of 2343 (4000x3000)images, divided into training (60%), validation (20%), and test (20%) sets. Each image set contains the original images alongside their groundtruth masks. The mask images will appear black as they are the ground truth pixel segmentation which contain the pixel values from range  n:number of classes so when displayed in general RGB it expects pixels range n:255


Floodnet Data
|    |    Train    |    Test    |    Validation    |
|----|-------------|------------|-----------------|
|    | train-original-images | train-original-images | validation-original-images |
|    | train-label-images (masks) | train-label-images (masks) | validation-label-images (masks) |

## PreProcessing
* Due to image dimensions being (4000x3000), I cropped the original images into 5780 (256256) of dimensions (256,256) as opposed to resizing in order to maintain accuracy on clear images. 

* Normalize train, test, and validation image sets through mean division as the pixel values range from 0 to 256 and normalizing will set between 0-1

* PSPNet model requires mask pixel values to be categorical due to the `Categorical _focal_ jaccard` loss function. Using Keras `to_categorical` on train, test, and validation masks

* No mask transformations needed for U-net model implementation

## Models


### U-Net
* U-shaped deep learning semantic segmentation model involving dividing the input image into smaller regions based on the objects and features present in the image. 


![image](https://user-images.githubusercontent.com/117116368/218929709-9f9fd4b7-d431-47fd-9902-79c26a52cea5.png)

* The U-Net model consists of an encoder, a skip connection, and decoder. The encoder downsamples the input using a combination of convolutional layers and max pooling method. The decoder upsamples using convolution layers to generate a segmentation map, and the skip connection uses the information from the corresponding encoding layer and reuses the features extracted to improve segmentation accuracy by preserving finer details.

* The final layer is the convolution layer containing activation function corresponding to the task as hand, which in this case was `softmax` for multiclass pixel classification and the number of classes for segmentation.

* Final model contained 23 convolultion layers with `SpatialDropout2D()` and  `BatchNormalization()` nested within the convolution_operation block. Below is the loss over 15 epochs performs well with around 70% accuracy with test and validation set.

![unet loss](https://user-images.githubusercontent.com/117116368/219537476-39fa4818-3b57-4598-ae35-ba9830386708.png)|![unet IoU](https://user-images.githubusercontent.com/117116368/219537571-b625f54b-f9b0-4bad-8242-3b3e9a29573a.png)

* Below are the U-Net mask predictions correctly segmenting a flooded building and flooded road.

![UNET FLOODED](https://user-images.githubusercontent.com/117116368/219533370-92363c87-0b32-4679-b160-4e67fe1f148d.png)

![UNet_idk](https://user-images.githubusercontent.com/117116368/219532661-5c5dff0c-5d94-47bd-9d54-cc2a984def04.png)

### Pyramid Scene Parsing Network(PSPNet)

![image](https://user-images.githubusercontent.com/117116368/219541459-a7478414-c94a-4eab-8d4f-aca725425fb1.png)
* As the U-Net model does its job well and with further tuning can potentially increase accuracy, the PSPNet model has been known to retain further important spatial features by using a pyramid style pooling method before concatenating and applying backbones. PSPNet is particularly useful for more complex/cluttered scenes where relationships between objects and their surroundings for example distinguish body of water v. flooded roads.


* Differently from the previous model the training, validation, and test masks need to be changed to categorical when fitting and predicting and use Intersection Over Union evaluation metric and Categorical Focal Jaccard Loss function.


* Best hyperparameters were with `dropout(0.5)`, `learning_rate(0.0001)`, and 8 epochs, while other attempts including learning rate of 0.0006, resnet101 backbones, and variating epochs.

![PSP_loss](https://user-images.githubusercontent.com/117116368/219543272-75fba946-5518-461d-b5a4-44d371b0a4f2.png)|![PSP_Iou](https://user-images.githubusercontent.com/117116368/219543291-ee0a3a14-439d-4a32-9c0f-deb2bb6ba032.png)

* When evaluated on the test set IoU was on average 30% with predicted test masks below.

![download (3)](https://user-images.githubusercontent.com/117116368/219696313-8b7790cf-c341-4f9b-864a-de8d8f127c17.png)

![PSPNet NonFlooded](https://user-images.githubusercontent.com/117116368/219544586-9386a6c7-1918-4771-8cf5-0672dc955505.png)

* Despite PSPNet not displaying better evaluation metrics than U-Net it does perform better in being able to correctly segment smaller objects like cars v. pools and correctly segments buildings on both flooded and non-flooded

## Conclusion 
* Natural disasters maintain prevalence today and with the use of UAV images these models can deploy effective response plans for highly effective areas in the most cost effective way.

* Both U-Net and PSPNet are effective models for semantic segmentation but with finer tuning, memory, and patience PSPNet should perform better than U-Net in maintaining spatial details as its pooling method produces more robust masks.

## References
[Original Paper](http://cs231n.stanford.edu/reports/2022/pdfs/21.pdf)

[Dataset](https://competitions.codalab.org/competitions/30290#participate)

## Contributions

All contributions are welcome. Please open an issue or create a pull request if you want to contribute to the project.


