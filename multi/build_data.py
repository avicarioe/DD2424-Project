from imutils import paths
import argparse
import random
import shutil
import os
from tqdm import tqdm

random.seed(0)

dataPath = "data"
datasetPath = "dataset2"
txtPath = "COVID-Net"

if os.path.exists(datasetPath):
    print("Error: dataset already exists")
    quit(1)

trainPath = os.path.sep.join([datasetPath, "train"])
testPath = os.path.sep.join([datasetPath, "test"])

def copy_img(txt, imagePath, outPath):
    os.makedirs(os.path.sep.join([outPath, "covid"]))
    os.makedirs(os.path.sep.join([outPath, "pneumonia"]))
    os.makedirs(os.path.sep.join([outPath, "normal"]))

    n_normal = 0
    n_covid = 0
    n_pneumonia = 0

    with open(txt) as f:
        lines = f.readlines()

    for l in tqdm(lines):
        l = l.strip()
        line = l.split(' ')
        img = os.path.sep.join([imagePath, line[1]])

        if not os.path.exists(img):
            print("Missing file:", img)
            continue

        if line[2] == "normal":
            n_normal += 1
            folder = "normal"
        elif line[2] == "pneumonia":
            n_pneumonia += 1
            folder = "pneumonia"
        elif line[2] == "COVID-19":
            n_covid += 1
            folder = "covid"
        else:
            print("Unk")
            continue

        outimg = os.path.sep.join([outPath, folder, line[1]])

        shutil.copy2(img, outimg)

    print("Normal:", n_normal)
    print("Pneumonia:", n_pneumonia)
    print("Covid:", n_covid)



traintxt = os.path.sep.join([txtPath, "train_COVIDx3.txt"])
traindata = os.path.sep.join([dataPath, "train"])
trainout = os.path.sep.join([datasetPath, "train"])

print("Train:")
copy_img(traintxt, traindata, trainout)

testtxt = os.path.sep.join([txtPath, "test_COVIDx3.txt"])
testdata = os.path.sep.join([dataPath, "test"])
testout = os.path.sep.join([datasetPath, "test"])

print("Test:")
copy_img(testtxt, testdata, testout)

print("Done")
