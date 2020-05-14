import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
 
model_path = "multi.model"
test_path = "dataset/test"

model = load_model(model_path)
model.summary()

imagesize = 224
batch_size = 32

testAug = ImageDataGenerator();

test = testAug.flow_from_directory(
    test_path,
    batch_size=batch_size,
    shuffle=False,
    target_size=(imagesize, imagesize))


class_weights = class_weight.compute_class_weight('balanced',
        np.unique(test.classes), test.classes)
class_weights = dict(enumerate(class_weights))

print(test.class_indices)
print(class_weights)

print("Evaluate network")
score = model.evaluate(x=test,
        batch_size=batch_size,
        steps=len(test),
        verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

test.reset()
predIdxs = model.predict(x=test,
        batch_size=batch_size,
        steps=len(test),
        verbose=1)
predIdxs = np.argmax(predIdxs, axis=1)

classes = test.classes[test.index_array]
print(sum(predIdxs==classes)/len(classes))

print(classification_report(classes, predIdxs, target_names=test.class_indices))

# Compute accuracy, sensitivity, and specificity
cm = confusion_matrix(classes, predIdxs)
print("Confusion matrix")
print(cm)
