import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.io
import matplotlib.pyplot as plt
import copy
from numpy import random
import numpy as geek
# import cv2
from PIL import Image,ImageOps
import math


##### load Im and label from the file

def pathname_folder(path):
    ### retunr the folder_path + filname
    return [path+f for f in listdir(path) if isfile(join(path, f))]

def filename_folder(path):
    ### retunr the folder_path + filname
    return [f for f in listdir(path) if isfile(join(path, f))]

# def original_mat(pathname):
#     ### load original image from the path
#     Im = scipy.io.loadmat(pathname)['Im']
#     Im = np.expand_dims(Im, axis=-1)  
#     return Im

def original_mat(pathname):
    data = scipy.io.loadmat(pathname)
    key = 'Im' if 'Im' in data else 'IM' if 'IM' in data else None
    if key is None:
        raise KeyError("Neither 'Im' nor 'IM' found in the .mat file.")
    Im = np.expand_dims(data[key], axis=-1)
    return Im
    

def heatmap_mat(pathname):
    BP = scipy.io.loadmat(pathname)['BT']
    BP = np.expand_dims(BP, axis=-1)  
    return BP

def array_where(condition):
    points_num = (np.where(condition))
    tupe_to_array = np.transpose(np.asarray(points_num))
    return tupe_to_array

def pad_image_symmetric(img,pad_size,pad_mode):
    ### return 4 dimensional symmetric padded from 3 dimensional original image
    if pad_mode == 'symmetric':
        img_pad = np.pad(img, pad_width=((pad_size[0], pad_size[0]), 
                                     (pad_size[1], pad_size[1]), 
                                     (pad_size[2], pad_size[2])), 
                     mode='symmetric')
     
    elif pad_mode == 'constant':
        img_pad = np.pad(img, pad_width=((pad_size[0], pad_size[0]), 
                                     (pad_size[1], pad_size[1]), 
                                     (pad_size[2], pad_size[2])), 
                     mode='constant',constant_values=((0,0),(0,0),(0,0)) )
        
    img_pad = np.expand_dims(img_pad, axis=-1) 
    return img_pad

def pad_label_symmetric(img,pad_size,pad_mode):
    ### return 4 dimensional symmetric padded from 4 dimensional labled image
    BP = pad_image_symmetric(img[:,:,:,0],pad_size,pad_mode)
    TP = pad_image_symmetric(img[:,:,:,1],pad_size,pad_mode)
    img_pad = np.concatenate((BP, TP), axis=-1)
    return img_pad

def each_dimension_random_number(crop_ID_array,d1,input_size,d2):
### to produce random shift among one dimension crop_ID_array[:,d1] to create positive crop_ID
### the shifted_size cannot exceed the input_size[d2]
### the shifted_size must larger than the crop_ID_array[:,d1]
    size = crop_ID_array.shape[0]
    if d2 < 2:
        random_min = (math.sqrt(2)-1)/2 * input_size[d2] 
        random_max = (math.sqrt(2)+1)/2 * input_size[d2]    
    else:
        random_min = 1
        random_max = input_size[d2]-1
        
    r = np.random.uniform(random_min, random_max,size = (size,1))
    
    
    ind = np.logical_and(crop_ID_array[:,d1] < input_size[d2],crop_ID_array[:,d1] > 0)
    # r[ind,0] = np.random.randint(0,list(crop_ID_array[ind,d1]))
    r[ind,0] = list(crop_ID_array[ind,d1] / 2)

    return r.astype(int)

def random_crop_id( crop_ID_array,input_size):
### it works with crop_ID_array.shape[0] > 1
    r_0 = each_dimension_random_number(crop_ID_array,0,input_size,0)
    r_1 = each_dimension_random_number(crop_ID_array,1,input_size,1)
    r_2 = each_dimension_random_number(crop_ID_array,2,input_size,2)
    random_crop_id = np.concatenate((r_0,r_1,r_2), axis=-1)
        
    return random_crop_id



# def random_crop_id( points_ID_array,input_size):
#     if len(points_ID_array.shape)==1:
#         size = 1
#         r_0 = np.random.randint(input_size[0]-2)
#         r_1 = np.random.randint(input_size[1]-2)
#         r_2 = np.random.randint(input_size[2]-2)
#         random_crop_id = np.array([r_0,r_1,r_2])
        
#     else:
#         size = points_ID_array.shape[0]
#         r_0 = np.random.randint(input_size[0]-2,size = (size,1))
#         r_1 = np.random.randint(input_size[1]-2,size = (size,1))
#         r_2 = np.random.randint(input_size[2]-2,size = (size,1))
#         random_crop_id = np.concatenate((r_0,r_1,r_2), axis=-1)
        
#     return random_crop_id

# def random_crop_id( ID_array,input_size):
    
#     for i in range(len(ID_array)):
#         np.random.randint((ID_array[0],input_size[2]-2)
        
    
    
    