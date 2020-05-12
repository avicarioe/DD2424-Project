import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# parameters:
LR = 1e-3
n_epoch = 10
batch_size = 8

datasetPath = "dataset"
print("Loading images..")
imagePaths = list(paths.list_images(datasetPath))
data = []
labels = []

for p in imagePaths:
    # Get label
    label = p.split(os.path.sep)[-2]

    # Resize and fix color
    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)

# One-hot encoding (index 0:= covid ,1 := noncovid)
data = np.array(data) / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# split into sets.train: 0.8, test: 0.1, validate: 0.1 
trainX, testX, trainY, testY = train_test_split(
    data, labels, test_size=0.10, stratify=labels, random_state=0)
trainX, valX, trainY, valY = train_test_split(
    trainX, trainY, test_size=1/9, stratify=trainY, random_state=0)

onehot = [i.argmax() for i in trainY]
class_weights = class_weight.compute_class_weight('balanced',np.unique(onehot),onehot)


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
    shear_range=2.0,
    )
# Load the imageNet weights
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Build the train model
headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = MaxPooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(lr=LR, decay=LR / n_epoch)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train
print("Training head")
H = model.fit(
    x=trainAug.flow(
        trainX,
        trainY,
        batch_size=batch_size,
        ),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(valX, valY),
    validation_steps=len(testX) // batch_size,
    epochs=n_epoch,
    class_weight = class_weights
)

print("Evaluate network")
predIdxs = model.predict(testX, batch_size=batch_size)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

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
fig.savefig("aug_plot.pdf", format="pdf", dpi=300, trasparent=True)
model.save("aug_covid.model", save_format="h5")
