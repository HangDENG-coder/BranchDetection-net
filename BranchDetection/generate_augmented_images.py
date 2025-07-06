import numpy as np
import sys
import os
from os import listdir
from os.path import isfile, join
import scipy.io
import matplotlib.pyplot as plt
import copy
from numpy import random
import numpy as geek
from PIL import Image,ImageOps
import math
import cv2
import taichi as ti
import time
import imutils
import pickle

import ultis
import transform_taichi
import load_image
from parameters import *
from tqdm import tqdm




print("Package Loaded, begin process!\n")
sys.stdout.flush()

### parameters
file_ind = -1

max_stacks = []

################################################################################################################
#   Imagine size calculation
################################################################################################################
def save_arr_pkl(filename,arr):
    with open(filename,'wb') as f:
        x = pickle.dump(arr,f)
    f.close()
    
def load_arr_pkl(filename):
    with open(filename,'rb') as f:
        arr = pickle.load(f)
    f.close()
    return arr

for file_ind in range(len(os.listdir(in_path))):
    img = load_image.Im_pad(in_path,pad_size,file_ind)
    max_stacks.append(img.shape[2])
num = np.max(max_stacks)
[x_size,y_size] = img.shape[0:2]



def img_augmentation(img,index):
    [file_ind,reverse,flip,deform,rotate_angle] = index
    reverse = int(reverse) % 2
    flip = int(flip) % 2
    deform = int(deform) % 5
    rotate_angle = int(rotate_angle) % 365
    img_aug = transform_taichi.Rotate_any_degree(transform_taichi.deform_3D(transform_taichi.Flip(transform_taichi.Reverse(img,reverse),flip),deform),rotate_angle)    
    return img_aug



tic = time.time()
for file_ind in range(len(max_stacks)):
    for i in range(2): ## reverse
        for j in range(2): ## flip
            for k in range(5): ## deform
                Im_pad_aug = np.zeros((1,1,1,1,x_size,y_size,num,1))
                Im_pad = load_image.Im_pad(in_path,pad_size,file_ind)
                Im_pad_aug[:,:,:,:,:,:,0:Im_pad.shape[2],:] = img_augmentation(Im_pad,[file_ind,i,j,k,0])  
                filename = Im_path +str(file_ind) +str(i)+str(j)+str(k)+'.pkl'
                save_arr_pkl(filename,Im_pad_aug)
                print([file_ind,i,j,k])
                sys.stdout.flush()
print("Im_completed")
toc = time.time()
print(toc-tic)




tic = time.time()
for file_ind in range(len(max_stacks)):
    for i in range(2): ## reverse
        for j in range(2): ## flip
            for k in range(5): ## deform
                label_img_pad_aug = np.zeros((1,1,1,1,x_size,y_size,num,2))
                label_img_pad = load_image.label_img_pad(in_path,file_ind,BP_path,TP_path,pad_size)
                label_img_pad_aug[:,:,:,:,:,:,0:label_img_pad.shape[2],:] = img_augmentation(label_img_pad,[file_ind,i,j,k,0])   
                filename = label_path+str(file_ind) +str(i)+str(j)+str(k)+'.pkl'
                save_arr_pkl(filename,label_img_pad_aug)
                print([file_ind,i,j,k])
                
                 
print("label_img_completed")
toc = time.time()
print(toc-tic)



tic = time.time()
for file_ind in range(len(max_stacks)):
    for i in range(2): ## reverse
        for j in range(2): ## flip
            for k in range(5): ## deform
                label_points_pad_aug = np.zeros((1,1,1,1,x_size,y_size,num,2))
                label_points_pad = load_image.label_points_pad(in_path,file_ind,BP_path,TP_path,pad_size)
                label_points_pad_aug[:,:,:,:,:,:,0:label_points_pad.shape[2],:] = img_augmentation(label_points_pad,[file_ind,i,j,k,0])     
                filename = points_path+str(file_ind) +str(i)+str(j)+str(k)+'.pkl'
                save_arr_pkl(filename,label_points_pad_aug)
                print([file_ind,i,j,k])

                
print("label_points_completed")
toc = time.time()
print(toc-tic)




tic = time.time()
for file_ind in tqdm(range(len(max_stacks))):
    for i in range(2): ## reverse
        for j in range(2): ## flip
            for k in range(5): ## deform
                Weights_pad_aug = np.zeros((1,1,1,1,x_size,y_size,num,1))
                Weights_pad = load_image.Im_pad(trace_path,pad_size,file_ind)
                Weights_pad_aug[:,:,:,:,:,:,0:Weights_pad.shape[2],:] = img_augmentation(Weights_pad,[file_ind,i,j,k,0])  
                filename = weights_path +str(file_ind) +str(i)+str(j)+str(k)+'.pkl'
                save_arr_pkl(filename,Weights_pad_aug)
                print([file_ind,i,j,k])
                sys.stdout.flush()
print("Weights_completed")
toc = time.time()
print(toc-tic)


print('Test Done!\n')









