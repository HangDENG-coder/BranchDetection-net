import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.io
import matplotlib.pyplot as plt
import copy
from numpy import random
import numpy as geek
import cv2
from PIL import Image,ImageOps
import math
import taichi as ti
import time
import ultis
import imutils
#######################################################################################
#### this Reverse works with gray-scale 3d stack of image for 3 dimension or 4 dimension


def Reverse(img,ind):  
    ind = ind%1
    if ind == 0:
        img_reverse = copy.deepcopy(img)
    elif ind == 1:
        if img.ndim == 3:
            img_reverse = geek.flip(img,axis = -1)
        elif img.ndim == 4:
            img_reverse = geek.flip(img,axis = -2)
    return img_reverse

#######################################################################################
#### this Flip works with gray-scale 3d stack of image for 3 dimension or 4 dimension

def Flip(img,ind):
    ind = ind%1
    if ind == 0:
        img_flip = copy.deepcopy(img)
    elif ind ==1:
        img_flip = np.flip(img, 1)
    return img_flip

#######################################################################################
#### this rotate works with gray-scale 3d stack of image for 3 dimension or 4 dimension
#### this rotate works with gray-scale 3d stack of image for  4 dimension in any degree
def Rotate_any_degree(img,angle):
#### this function rotate image(x,y,z,channel) for angle degrees
    angle = angle % 365
    img_rotated = np.zeros(img.shape)
    for j in range(img.shape[3]):
        for i in range(img.shape[2]):
            img_rotated[:,:,i,j] = imutils.rotate(img[:,:,i,j], angle=angle)
    return  img_rotated 


#################################################################################################################################

ti.init(arch=ti.cpu)

#######################################################################################
################# waveform_multidirectional ######################################
@ti.kernel
def waveform_multidirectional(img: ti.types.ndarray(), img_output: ti.types.ndarray()):
### work for 2d gray-scale image
    rows, cols = img.shape
    for i,j in img_output:            
        offset_x = int(20.0 * ti.sin(2 * 3.14 * i / 150))
        offset_y = int(20.0 * ti.cos(2 * 3.14 * j / 150))
        if i+offset_y < rows and j+offset_x < cols:
            x = int((i+offset_y)%rows)
            y = int((j+offset_x)%cols)
            img_output[i,j] += img[x,y]
        else:
            img_output[i,j] += 0
    
    
def taichi_waveform_multidirectional(img):
    img_output = np.zeros(img.shape)
    waveform_multidirectional(np.ascontiguousarray(img),img_output)
    return img_output

def taichi_waveform_multidirectional_3D(img):
    img_out = np.zeros(img.shape)
    for j in range(img.shape[3]):
        for i in range(img.shape[2]):
            img_out[:,:,i,j] = taichi_waveform_multidirectional(img[:,:,i,j])
    
    return img_out

#######################################################################################
################# waveform_vertical ######################################
@ti.kernel
def waveform_vertical(img: ti.types.ndarray(), img_output: ti.types.ndarray()):
### work for 2d gray-scale image
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            offset_x = int(25.0 * ti.sin(2 * 3.14 * i / 180))
            offset_y = 0
            if j+offset_x < rows:
                img_output[i,j] = img[i,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0
                
def taichi_waveform_vertical(img):
    img_output = np.zeros(img.shape)
    waveform_vertical(np.ascontiguousarray(img),img_output)
    return img_output

def taichi_waveform_vertical_3D(img):
    img_out = np.zeros(img.shape)
    for j in range(img.shape[3]):
        for i in range(img.shape[2]):
            img_out[:,:,i,j] = taichi_waveform_vertical(img[:,:,i,j])
    
    return img_out                
                
#######################################################################################
################# waveform_horizontal ######################################
@ti.kernel
def waveform_horizontal(img: ti.types.ndarray(), img_output: ti.types.ndarray()):
### work for 2d gray-scale image
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            offset_x = 0
            offset_y = int(16.0 * ti.sin(2 * 3.14 * j / 150))
            if i+offset_y < rows:
                img_output[i,j] = img[(i+offset_y)%rows,j]
            else:
                img_output[i,j] = 0

def taichi_waveform_horizontal(img):
    img_output = np.zeros(img.shape)
    waveform_horizontal(np.ascontiguousarray(img),img_output)
    return img_output

def taichi_waveform_horizontal_3D(img):
    img_out = np.zeros(img.shape)
    for j in range(img.shape[3]):
        for i in range(img.shape[2]):
            img_out[:,:,i,j] = taichi_waveform_horizontal(img[:,:,i,j])
    
    return img_out    

#######################################################################################
################# waveforms concave ######################################
@ti.kernel
def deform_concave(img: ti.types.ndarray(), img_output: ti.types.ndarray()):
### work for 2d gray-scale image
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            offset_x = int(128.0 * ti.sin(2 * 3.14 * i / (2*cols)))
            offset_y = 0
            if j+offset_x < cols:
                img_output[i,j] = img[i,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0
                
def taichi_deform_concave(img):
    img_output = np.zeros(img.shape)
    deform_concave(np.ascontiguousarray(img),img_output)
    return img_output

def taichi_deform_concave_3D(img):
    img_out = np.zeros(img.shape)
    for j in range(img.shape[3]):
        for i in range(img.shape[2]):
            img_out[:,:,i,j] = taichi_deform_concave(img[:,:,i,j])
    
    return img_out                    


#######################################################################################
################# five waveforms ######################################
def deform_3D(img,ind):
    ind = ind % 5
    if ind == 0:
        img_deform = copy.deepcopy(img)
    elif ind == 1:
        img_deform = taichi_waveform_vertical_3D(img)
    elif ind == 2:
        img_deform = taichi_waveform_horizontal_3D(img)
    elif ind == 3:
        img_deform = taichi_waveform_multidirectional_3D(img)
    elif ind == 4:
        img_deform = taichi_deform_concave_3D(img)
    return img_deform





