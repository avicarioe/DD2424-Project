import numpy as np
import os
import cv2
import random

from imutils import paths
from tqdm import tqdm

import matplotlib.pylab as plt

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Conv2D
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

n_epoch = 15
batch_size = 16
LR = 1e-5
imagesize = 224

train_path = 'dataset2/train'
test_path='dataset2/test'
outname = "x_inception"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)


# Initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=5,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1,
    # brightness_range=(0.95,1.05),
    zoom_range=0.2,
    horizontal_flip=1,
    # channel_shift_range=30.0,
    shear_range=2.0)


train = trainAug.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=(imagesize, imagesize),
    shuffle=True)

class_weights = class_weight.compute_class_weight('balanced', np.unique(train.classes), train.classes)
class_weights = dict(enumerate(class_weights))

print(class_weights)


num_classes = train.num_classes


## Load the imageNet weights
baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(imagesize, imagesize, 3)))
#baseModel = InceptionV3(weights='imagenet', include_top=False, 
#        input_tensor=Input(shape=(imagesize, imagesize,3)))
headModel = baseModel.output

#headModel = MaxPooling2D(pool_size=(3, 3))(headModel)
headModel = Conv2D(100, kernel_size = (3,3), padding = 'valid')(headModel)
headModel = Flatten()(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(num_classes, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Train top layer
for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(lr=LR, decay=LR / n_epoch)
model.compile(loss='categorical_crossentropy', 
        optimizer=opt, metrics=['accuracy'])
model.summary()

H = model.fit(
        x=train,
        epochs=n_epoch, 
        steps_per_epoch=len(train),
        batch_size=batch_size,
        class_weight=class_weights)

print("Save model")
model.save(outname + "_model.model", save_format="h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
fig = plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
fig.set_size_inches((7, 6))
fig.savefig(outname + "_plot.pdf", format="pdf", dpi=300, trasparent=True)
