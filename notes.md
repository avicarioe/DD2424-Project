# Deep learning Project:

### Effects of Domain Adaptation on Performance of Transfer Learning on COVID-19 X-ray Images

These are notes written for internal use. 





## Todo list

- [ ] gradcam article read

- [ ] understand ***pulmonary edemas*** relation to pneumonia and covid xray data

- [ ] 

  



## Approach

Our efforts can be separated into three general tasks. We can then compare them and their successes. 

* Load ImageNet (VGGNet?) weights and apply transfer learning to covid-19 images
* Train our own model for pneumonia classifications on x-ray data. Thereafter transfer the learning to COVID 19 cases.
  * Are there pretrained pneumonia classifier models available?
  * Viable network architecture? 
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

* info

#### [Grad-CAM: Visualize class activation maps with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/)

* info 