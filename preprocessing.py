# Author: Calebe Monteiro
# Reference: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
#
import numpy as np
import pandas as pd
#
import cv2
import os
#
import random
#
# Load input file
curdir = os.getcwd()
filename = "\\data\\fer2013.csv"
data = pd.read_csv(curdir+filename,delimiter=',',dtype='a')
#
# Define Dict for dataset split
d1 = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral" }
d2 = {"PrivateTest": "Validation", "PublicTest": "Validation", "Training": "Training" }
#
# Define the labels
dirs = set(data['emotion'])
labels = np.array(data['emotion'],np.float)
imagebuffer = np.array(data['pixels'])
#
# Function for unique filename
def unique_name(pardir,prefix,suffix='jpg'):
    filename = '{0}_{1}.{2}'.format(prefix,random.randint(1,10**8),suffix)
    filepath = os.path.join(pardir,filename)
    if not os.path.exists(filepath):
        return filepath
    unique_name(pardir,prefix,suffix)      
#
# Defines the array "images" from the np.array "imagebuffer" as unsigned integer and deletes the original 
# imagebuffer for memory saving.
images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer])
del imagebuffer
#
# Calculate the square root of the images tuple
num_shape = int(np.sqrt(images.shape[-1]))
images.shape = (images.shape[0],num_shape,num_shape)
#
# Calculate training set and validation sets sizes
#training_size= round(len(data) / 80 * 10)
#validate_size = len(data) - training_size
#
# Prepare folders/subfolders for the Dataset Split
class_dir = {}
for dr in dirs:
#
#   Create Training Folders
    dest = os.path.join(curdir + "\\Training\\", d1[int(dr)])
    class_dir[dr] = dest

    if not os.path.exists(dest):
        os.makedirs(dest)
#
#   Create Validation Folders
    dest = os.path.join(curdir + "\\Validation\\", d1[int(dr)])
    class_dir[dr] = dest
#
    if not os.path.exists(dest):
        os.makedirs(dest)
#
data = zip(labels,images,data['Usage'])
#
# Main loop for image generation
for d in data:
    destdir = os.path.join(curdir,d2[d[2]],d1[int(d[0])])
    filepath = unique_name(destdir,d2[d[2]])
    img = d[1]
    if not os.path.exists(destdir):
        os.mkdir(destdir)
    if not filepath:
        continue
    sig = cv2.imwrite(filepath,img)
    print('[^_^] Write image to %s' % filepath)
    if not sig:
        print('Error')
        exit(-1)

print('---------------------------')
print(' Training Dataset ')
print('---------------------------')
for expression in os.listdir(curdir + "\\Training\\"):
    print(str(len(os.listdir(curdir + "\\Training\\" + expression))) + " " + expression + " images")
print('---------------------------')

print('---------------------------')
print(' Validation Dataset ')
print('---------------------------')
for expression in os.listdir(curdir + "\\Validation\\"):
    print(str(len(os.listdir(curdir + "\\Validation\\" + expression))) + " " + expression + " images")
print('---------------------------')