# Post-Disaster-Semenatic-Segmentation

![download (11)](https://user-images.githubusercontent.com/117116368/218915983-da8ef616-fc88-49b6-b031-90f7faa0aa9e.png)
## Table of Contents

* [Overview](#Overview)
* [Requirements](#Requirements)
* [Dataset](#Dataset)
* [Refrences](#References)
* [Contributions](#Contributions)

## Overview
This repository contains code for a deep learning based semantic segmentation model for scene parsing of high-quality, low-resolution, post-disaster UAV imagery analysis. The model is trained to classify 10 different classes including Background, Building-Flooded, Building-NonFlooded,Road, Road-NonFlooded, Tree, Pool, Water, Vehicle, and Grass of disaster stricken areas. This project is motivated by the increasing frequency and severity of floods, which can cause widespread damage to infrastructure and disrupt lives. In order to effectively respond to natural disasters, it is important to quickly assess the extent of damage and identify areas that need priority attention.

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
The original dataset consists of 2,343 , (4000,3000) -Dimension images, divided into training (60%), validation (20%), and test (20%) sets. Each image set contains the original images alongside their groundtruth mask images. The mask images will appear black as they are the ground truth pixel segmentation which contain the pixel values from range  n:number of classes so when displayed in general RGB it expects pixels range n:255


Floodnet Data
|    |    Train    |    Test    |    Validation    |
|----|-------------|------------|-----------------|
|    | train-original-images | train-original-images | validation-original-images |
|    | train-label-images (masks) | train-label-images (masks) | validation-label-images (masks) |

## PreProcessing
* Due to image dimensions being of height =4000, width= 3000, I cropped the orginal images into ____ check when....... of dimmensions (256,256) as opposed to resizing in order to maintain accuracy on clear images. 

* Normalize train, test, and validation image sets through mean division as the pixel values range from 0 to 256 and normalizing will set bewteen 0-1

* PSPNet model requires mask pixel values to be categorical due to the `Categorical _focal_ jaccard` loss funcation. Using Keras `to_categorical` on train, test, and validation masks

* No mask transformations needed for U-net model implimintation

## Models


### U-Net
* U-shaped like deep learning semantic segmenatation model involving dividing the input image into smaller regions based on the objects and features present in the image. 


![image](https://user-images.githubusercontent.com/117116368/218929709-9f9fd4b7-d431-47fd-9902-79c26a52cea5.png)

* The U-Net model consist of an encoder, a skip connection, and decoder. The encoder downsamples the input using a combination of convolutional layers and maxpooling method. The decoder upsamples using convolution layers to generate a segmenatation map, and the skip connection uses the information from the corresponding encoding layer and reuses the features extracted to improve segmenatation accuracy by preserving finer details.

* The final layer is the convolution layer containing activation function corresponding to task as hand, which in this case was `softmax` for multiclass pixel classification




## References
[Original Paper](http://cs231n.stanford.edu/reports/2022/pdfs/21.pdf)

[Dataset](https://competitions.codalab.org/competitions/30290#participate)

## Contributions

All contributions are welcome. Please open an issue or create a pull request if you want to contribute to the project.
