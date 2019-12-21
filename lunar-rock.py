#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 17:39:11 2019

@author: arshita
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import mean,median

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def rgb_to_bin(path):
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 30, 30), (75, 255,255))
    return mask

def number_of_contours(mas):
    cnts = cv2.findContours(mas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    total = 0
    pixel = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], [255,255,255])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pixels = cv2.countNonZero(mask)
        pixel.append(pixels)
        total += pixels
        cv2.putText(image, '{}'.format(pixels), (x,y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    return(pixel)

size = []
Class = []
images = []
number = []

df = pd.read_csv("train.csv")
for i in range(len(df)):
    path = 'classification of lunar rock/DataSet/Train Images/' + df["Class"][i] + "/" + df["Image_File"][i]
    mas = rgb_to_bin(path)
    
    pixel = number_of_contours(mas)
    
    number.append(len(pixel))
    images.append(df["Image_File"][i])
    
    if df["Class"][i]=='Small':
        Class.append(0)
    else:
        Class.append(1)
    
    if len(pixel) > 0:
        size.append(median(pixel))
        #size.append(mean(pixel))
    elif len(pixel) == 0:
        size.append(0)

dict = {'Image_File':images, 'Class':Class, 'Number':number, 'Median-size':size}
data =  pd.DataFrame(dict)

size = []
images = []
number = []

df = pd.read_csv("test.csv")
for i in range(len(df)):
    path = 'classification of lunar rock/DataSet/Test Images/' + df["Image_File"][i]
    mas = rgb_to_bin(path)
    
    pixel = number_of_contours(mas)
    
    number.append(len(pixel))
    images.append(df["Image_File"][i])
    
    if len(pixel) > 0:
        size.append(median(pixel))
        #size.append(mean(pixel))
    elif len(pixel) == 0:
        size.append(0)

dict = {'Image_File':images, 'Number':number, 'Median-size':size}
data1 =  pd.DataFrame(dict)

#svm model
x = data.iloc[:,3:]
y = data["Class"]
x_test = data1.iloc[:,2:]

svc = SVC(kernel='linear').fit(x,y)
y_pred = svc.predict(x_test)

clas=[]
for i in y_pred:
    if i==0:
        clas.append("Small")
    else:
        clas.append("Large")
        
image=[]
for k in range(len(data1)):
    image.append(data1["Image_File"][k])

dict = {'Image_File':image, 'Class': clas}
dataframe =  pd.DataFrame(dict)
dataframe.to_csv("test.csv")