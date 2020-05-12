import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
 
# Load the image and change it into an array and expand the dimensions
# img = load_img('snail.jpg')
img = load_img('test_ray.jpg')
img = img_to_array(img)
img1 = np.expand_dims(img, axis=0)
 
# create an instance of the class with the desired operation
datagen = ImageDataGenerator(
    rotation_range=10,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8,1.2),
    zoom_range=0.2,
    horizontal_flip=1,
    channel_shift_range=50.0,
    shear_range=5.0,
    )
# Depending on the augmentation method you may need to call
# fit method to calculate the global statistics
data_generator = datagen.flow(img1,batch_size=1)
 
# Display some augmented samples
plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    for x in data_generator:
        plt.imshow(x[0]/255.)
        plt.xticks([])
        plt.yticks([])
        break
plt.tight_layout()
plt.show()