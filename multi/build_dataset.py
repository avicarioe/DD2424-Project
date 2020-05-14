import pandas as pd
from imutils import paths
import argparse
import random
import shutil
import os

random.seed(0)

kagglePath = "chest_xray"
covidPath = "covid-chestxray-dataset"
datasetPath = "dataset"
trainPath = os.path.sep.join([datasetPath, "train"])
valPath = os.path.sep.join([datasetPath, "val"])
testPath = os.path.sep.join([datasetPath, "test"])

val_size = 0.08
test_size = 0.15

if os.path.exists(datasetPath):
    print("Error: dataset already exists")
    quit(1)

# Load covid
os.makedirs(os.path.sep.join([trainPath, "covid"]))
os.makedirs(os.path.sep.join([valPath, "covid"]))
os.makedirs(os.path.sep.join([testPath, "covid"]))

csvPath = os.path.sep.join([covidPath, "metadata.csv"])
df = pd.read_csv(csvPath)

covidImages = []

for (i, row) in df.iterrows():
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    imagePath = os.path.sep.join([covidPath, "images", row["filename"]])

    if not os.path.exists(imagePath):
        continue

    covidImages.append(imagePath)

n_val = int(len(covidImages) * val_size)
n_test = int(len(covidImages) * test_size)
n_train = len(covidImages) - n_val - n_test

random.shuffle(covidImages)

covid_train = covidImages[:n_train]
covid_val = covidImages[n_train:n_train + n_val]
covid_test = covidImages[n_train + n_val:]

def copy_img(images, outPath):
    for im in images:

        # Construct the path to the copied image
        filename = im.split(os.path.sep)[-1]
        outputPath = os.path.sep.join([outPath, "covid", filename])

        #print(filename)

        shutil.copy2(imagePath, outputPath)

    print("Covid: ", len(images))

# Load kaggle


intrainPath = os.path.sep.join([kagglePath, "train"])
invalPath = os.path.sep.join([kagglePath, "val"])
intestPath = os.path.sep.join([kagglePath, "test"])

def copy_kaggle(inpath, outpath):
    os.makedirs(os.path.sep.join([outpath, "normal"]))
    os.makedirs(os.path.sep.join([outpath, "virus"]))
    os.makedirs(os.path.sep.join([outpath, "bacteria"]))

    normalPath = os.path.sep.join([inpath, "NORMAL"])
    pneumoniaPath = os.path.sep.join([inpath, "PNEUMONIA"])

    normalIm = list(paths.list_images(normalPath))
    pneumoniaIm = list(paths.list_images(pneumoniaPath))

    # Construct the path to the copied image
    for image in normalIm:
        filename = image.split(os.path.sep)[-1]
        outputPath = os.path.sep.join([outpath, "normal", filename])

        #print(filename)

        shutil.copy2(image, outputPath)

    n_virus = 0
    n_bacteria = 0

    for image in pneumoniaIm:
        filename = image.split(os.path.sep)[-1]

        if "bacteria" in filename:
            outputPath = os.path.sep.join([outpath, "bacteria", filename])
            n_bacteria += 1
        elif "virus" in filename:
            outputPath = os.path.sep.join([outpath, "virus", filename])
            n_virus += 1
        else:
            continue

        #print(filename)

        shutil.copy2(image, outputPath)

    print("Normal: ", len(normalIm))
    print("Bacteria: ", n_bacteria)
    print("Virus: ", n_virus)

print("Train:")
copy_img(covid_train, trainPath)
copy_kaggle(intrainPath, os.path.sep.join([datasetPath, "train"]))
print("Val:")
copy_kaggle(invalPath, os.path.sep.join([datasetPath, "val"]))
copy_img(covid_val, valPath)
print("Test:")
copy_kaggle(intestPath, os.path.sep.join([datasetPath, "test"]))
copy_img(covid_test, testPath)

print("Done")
