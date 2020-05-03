# Deep learning Project:

### Effects of Domain Adaptation on Performance of Transfer Learning on COVID-19 X-ray Images

These are notes written for internal use. 

Pneumonia is in an inflammation of the lung. (SARS-CoV-2) can also result in pneumonia, as it is a virus and in adults, viruses account for about one third of pneumonia cases.  [pulmonary edema](https://en.wikipedia.org/wiki/Pulmonary_edema) is excess fluids in the lung. Unclear if covid can cause pulmonary edma, if so both *pulmonary edema* and *pneumonia* can be byproducts of covid but not necessarily indicators? 



## Todo list

- [ ] gradcam article read

- [ ] is there enough data for non-COVID pneumonia cases? 

- [ ] transfer learning article read 

  

## Approach

Our efforts can be separated into three general tasks. We can then compare them and their successes. 

* Load ImageNet (VGGNet?) weights and apply transfer learning to covid-19 images
* Train our own model for pneumonia classifications on x-ray data. Thereafter transfer the learning to COVID 19 cases.
  * Are there pretrained pneumonia classifier models available?
  * Is there enough data for non-covid pneumonia cases? 
  * Viable network architecture? (how deep?)
* Preprocess and  augment the COVID data set: 
  * Open CV libraries 
  * What is the standard procedure to adapt pretrained models expecting RBG inputs to greyscale? 



## Resources

Below are some notes on the resources found along the way. 

#### [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)

* OpenCV for loading and preprocessing 
  * normalization
* one hot encoding 
* VGGNet model weights used. 500 Mb 
  * input with the size 224 x 224 pixels with 3 channels RGB
  * freeze the imported weights, train the last layer(s).
* reportedly 92 % accurate on small COVID19 dataset, $\approx 30$ images per class.
* critique: "I think it would be cool if you could rerun this experiment with three  test sets: healthy, COVID-19 positive, and pneumonia cases that are  COVID-19 negative! This would be a good control to see if the model is  <u>truly detecting COVID cases, or just detecting fluid in the lungs.</u>"



#### [Detecting COVID-19 induced Pneumonia from Chest X-rays with Transfer Learning: An implementation in Tensorflow and Keras.](https://towardsdatascience.com/detecting-covid-19-induced-pneumonia-from-chest-x-rays-with-transfer-learning-an-implementation-311484e6afc1)

*  [VGG16 network](https://arxiv.org/pdf/1409.1556.pdf) used for transfer learning 
* 2 dense layers trained with low learning rate 5e-4 for binary classification, 1E-4 for multi.  
* combining the [Kaggle Chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) with the [COVID19 Chest X-ray dataset](https://github.com/ieee8023/covid-chestxray-dataset): 
  * Healthy: 79 images
  * Pneumonia (Viral) : 79 images
  * Pneumonia (Bacterial): 79 images
  * Pneumonia (COVID-19): 69 images
  * 9 images / class in test sets.

* cleverly augmenting and improving these results for the multi-class case should be interesting. 
* original article makes no attempt to augment small datasets.

#### [Grad-CAM: Visualize class activation maps with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/)

* info 



#### [Improving Classification Accuracy using Data Augmentation & Segmentation: A  hybrid implementation in Keras & Tensorflow using Transfer Learning](https://medium.com/gradientcrescent/improving-classification-accuracy-using-data-augmentation-segmentation-a-hybrid-implementation-8ec29fa97043)

* base: MobileNetV2 model, with the top layers relaced with a two densely  connected layers (of size 1024 and 196, respectively), separated by a  50% dropout layer to prevent overfitting.
*  ADAM optimizer at a learning rate of 0.0002. 50 epochs 
* *YOLO is itself a InceptionV3-based trained neural network classifier,  and is designed to detect the primary subject of an image, generate a  boundary box for the subject, and then crop out the background.* 
* The Keras library allows us to utilize traditional physical augmentation methods, such as:
  * Rescaling
  * Shear-based transformations
  * Zoom-based transformations
  * Translations
  * Flipping

* Tensorflow library provides more advanced capabilities to modify the  colorspace:
  * Hue randomization
  * Saturation randomization
  * Brightness randomization
  * Contrast randomization