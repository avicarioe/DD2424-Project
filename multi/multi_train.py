import numpy as np
import os
import cv2
import random

from imutils import paths
from tqdm import tqdm

import matplotlib.pylab as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Conv2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

n_epoch = 5
batch_size = 32
LR = 1e-3

train_path = 'dataset/train'
val_path ='dataset/val'
test_path='dataset/test'

print("Loading data")
imagesize = 300

def loadImages(path):
    data = []
    labels = []
    for cat in os.listdir(path):
        if cat.startswith('.'):
            continue

        imagePath = os.path.sep.join([path, cat])
        print(imagePath)
        imageList = list(paths.list_images(imagePath))
        for im in tqdm(imageList):
            if im.startswith('.'):
                continue

            img = cv2.imread(im)
            img = cv2.resize(img, (imagesize, imagesize))

            labels.append(cat)
            data.append(img)

    return data, labels


encoder = LabelEncoder()

valX, valL = loadImages(val_path)
valX = np.array(valX) / 255.0
valL = np.array(valL)


valL = encoder.fit_transform(valL)
classes = list(encoder.classes_)
print(classes)
valY = to_categorical(valL, num_classes=len(classes))


#y_integers = np.argmax(trainY, axis=1)
#class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
#class_weights = dict(enumerate(class_weights))
#print(class_weights)

class_weights = {0: 1.3, 1: 0.45, 2: 15}

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

valAug = ImageDataGenerator();

## Load the imageNet weights
baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(imagesize, imagesize, 3)))
headModel = baseModel.output

headModel = MaxPooling2D(pool_size=(3, 3))(headModel)
#headModel = Conv2D(100, kernel_size = (3,3), padding = 'valid')(headModel)
headModel = Flatten()(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(classes), activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Train top layer
for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(lr=LR, decay=LR / n_epoch)
model.compile(loss='categorical_crossentropy', 
        optimizer=opt, metrics=['accuracy'])
model.summary()

train = trainAug.flow_from_directory(
    train_path,
    classes=classes,
    batch_size=batch_size,
    target_size=(imagesize, imagesize),
    shuffle=True)

val = valAug.flow_from_directory(
    val_path,
    classes=classes,
    batch_size=batch_size,
    target_size=(imagesize, imagesize))

H = model.fit(
        x=train,
        epochs=n_epoch, 
        steps_per_epoch=len(train),
        batch_size=batch_size,
        validation_data=val,
        validation_steps=len(val))
        #class_weight=class_weights)

print("Save model")
model.save("multi.model", save_format="h5")
fig.savefig("plot.pdf", format="pdf", dpi=300, trasparent=True)
