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
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

n_epoch = 5
batch_size = 8
LR = 1e-3

train_path = 'dataset/train'
val_path ='dataset/val'
test_path='dataset/test'

print("Loading data")
imagesize = 100

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
num_classes = 3

trainX, labels = loadImages(train_path)
trainX = np.array(trainX) / 255.0
labels = np.array(labels)

labels = encoder.fit_transform(labels)
trainY = to_categorical(labels, num_classes=num_classes)

valX, labels = loadImages(val_path)
valX = np.array(valX) / 255.0
labels = np.array(labels)

labels = encoder.fit_transform(labels)
valY = to_categorical(labels, num_classes=num_classes)

y_integers = np.argmax(trainY, axis=1)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
class_weights = dict(enumerate(class_weights))
print(class_weights)

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

## Load the imageNet weights
baseModel = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(imagesize, imagesize, 3)))
headModel = baseModel.output

headModel = MaxPooling2D(pool_size=(3, 3))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(num_classes, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Train top layer
for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(lr=LR, decay=LR / n_epoch)
model.compile(loss='categorical_crossentropy', 
        optimizer=opt, metrics=['accuracy'])
model.summary()

H = model.fit(x=trainAug.flow(trainX, trainY, batch_size=batch_size),
        epochs=n_epoch, 
        steps_per_epoch=len(trainX) // batch_size,
        batch_size = batch_size,
        validation_data=(valX, valY), 
        validation_steps=len(valX) // batch_size,
        class_weight=class_weights)

model.save("multi.model", save_format="h5")

print("Evaluate network")

testX, labels = loadImages(test_path)
testX = np.array(testX) / 255.0
labels = np.array(labels)

labels = encoder.fit_transform(labels)
testY = to_categorical(labels, num_classes=num_classes)

predIdxs = model.predict(testX, batch_size=batch_size)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs, target_names=encoder.classes_))

# Compute accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# Plot the training loss and accuracy
plt.style.use("ggplot")
fig = plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
fig.set_size_inches((7, 6))

fig.savefig("plot.pdf", format="pdf", dpi=300, trasparent=True)
