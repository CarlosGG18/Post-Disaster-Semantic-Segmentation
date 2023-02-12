# Post-Disaster-Semenatic-Segmentation

![image](https://user-images.githubusercontent.com/117116368/218328152-e555ef18-8662-41ba-a0aa-e798e19eb0f4.png)
## Table of Contents

* [Overview](#Overview)
* [Requirements](#Requirements)
* [Dataset](#Dataset)
* [Refrences](#References)
* [Contributions](#Contributions)

## Overview
This repository contains code for a deep learning based semantic segmentation model for scene parsing high-quality, low-resolution, post-disaster UAV imagery analysis. The model is trained to classify 10 different classes including Background, Building-Flooded, Building-NonFlooded,Road, Road-NonFlooded, Tree, Pool, Water, Vehicle, and Grass of disaster stricken areas. This project is motivated by the increasing frequency and severity of floods, which can cause widespread damage to infrastructure and disrupt lives. In order to effectively respond to natural disasters, it is important to quickly assess the extent of damage and identify areas that need priority attention.

The process of aerial image classification involves two stages: semantic segmentation and pixel-wise classification. Semantic segmentation breaks down an image into uniform regions, allowing us to understand the local structure of the image and distinguish between boundaries such as lines and curves. This process involves assigning a label to each pixel in the image, where pixels with the same label possess similar visual attributes.

## Requirements
The code in this repository is written in Python and requires the following packages:
* Segmentation-Models
* Tensorflow
* Keras
* Numpy
* OpenCV
* Matplotlib

## Dataset
The original dataset consists of 2,343 images, divided into training (~60%), validation (~20%), and test (~20%) sets. Each testset contains the original images alongside their groundtruth mask images. 


Floodnet Data
|    |    Train    |    Test    |    Validation    |
|----|-------------|------------|-----------------|
|    | train-original-images | train-original-images | validation-original-images |
|    | train-label-images (masks) | train-label-images (masks) | validation-label-images (masks) |

## References
[Original Paper](http://cs231n.stanford.edu/reports/2022/pdfs/21.pdf)

[Dataset](https://competitions.codalab.org/competitions/30290#participate)

## Contributions

All contributions are welcome. Please open an issue or create a pull request if you want to contribute to the project.
