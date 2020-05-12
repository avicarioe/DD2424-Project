import pandas as pd
from imutils import paths
import argparse
import random
import shutil
import os

kagglePath = "chest_xray"
basePath = os.path.sep.join([kagglePath, "train", "NORMAL"])
imagePaths = list(paths.list_images(basePath))

covidPath = "covid-chestxray-dataset"

datasetPath = "dataset"

if os.path.exists(datasetPath):
	print("Error: dataset already exists")
	quit(1)
os.makedirs(os.path.sep.join([datasetPath, "normal"]))
os.makedirs(os.path.sep.join([datasetPath, "covid"]))

csvPath = os.path.sep.join([covidPath, "metadata.csv"])
df = pd.read_csv(csvPath)

n_covid = 0

for (i, row) in df.iterrows():
	if row["finding"] != "COVID-19" or row["view"] != "PA":
		continue

	imagePath = os.path.sep.join([covidPath, "images",
		row["filename"]])

	if not os.path.exists(imagePath):
		continue

	# Construct the path to the copied image
	filename = row["filename"].split(os.path.sep)[-1]
	outputPath = os.path.sep.join([datasetPath, "covid", filename])

	print(filename)
	n_covid += 1

	shutil.copy2(imagePath, outputPath)


# randomly sample the image paths
random.seed(0)
random.shuffle(imagePaths)
n_kaggle = min(n_covid, len(imagePaths))
imagePaths = imagePaths[:n_kaggle]

# Construct the path to the copied image
for (i, imagePath) in enumerate(imagePaths):
	filename = imagePath.split(os.path.sep)[-1]
	outputPath = os.path.sep.join([datasetPath, "normal", filename])

	print(filename)

	shutil.copy2(imagePath, outputPath)


print("{} normal images".format(n_kaggle))

print("{} covid images".format(n_covid))
