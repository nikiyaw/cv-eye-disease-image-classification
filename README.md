# Image Classification, Predicting Eye Diseases

## Overview
The goal of this project is to train a model via image classification and use the trained model to accurately predict eye diseases. 

## Tools and Programming Language(s)
Python, Tensorflow, Keras, CNN

## Link to Kaggle dataset

https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

## Process
1. Obtain the eye disease image dataset from Kaggle.
2. Import all the necessary libraries.
3. Load the dataset according with the file paths and split the dataset accordingly.
4. Augment the image datasets into a format that can be processed and trained using Tensorflow and Keras.
5. Do a quick check and display the augmented images.
6. Download the CNN model (EfficientNetB3 architecture pre-trained on ImageNet) and compile it for training.
7. Fit the model with the augmented train dataset and augmented valid dataset.
8. Plot the loss curve analysis and accuracy curve analysis graphs.
9. Run the model and compare the actual and predicted images side by side.

## Acknowledgement
I would like to mention an article on Medium that provided the step-by-step tutorial for this project. 

https://medium.com/@deasadiqbal/computer-vision-project-image-classification-with-tensorflow-and-keras-264944d09721
