import numpy as np
import os
from os import listdir
from os.path import isfile, join, exists
import scipy.io
import matplotlib.pyplot as plt
import copy
from numpy import random
import numpy as geek
# import cv2
from PIL import Image,ImageOps
import math
import ultis
# import transform_taichi
import pickle

################################################################################################################################
###################################### Load original_image stacks and labeled_images from file #################################



def load_from_pkl(pkl_filename,key_name):
    if os.path.exists(pkl_filename):
        pkl_file = open(pkl_filename, 'rb')
        mydict = pickle.load(pkl_file)
        pkl_file.close()
        if key_name in list(mydict.keys()):
            return mydict[key_name]
    else:
        print("file doesn't exist")


################################################################################################################################
###################################### Load original_image stacks and labeled_images from file #################################

def Im_pad(in_path,pad_size,file_ind):
    onlyfiles = ultis.filename_folder(in_path)
    img_filename = onlyfiles[file_ind]
    
    Im = ultis.original_mat(in_path+img_filename)
    Im_pad = ultis.pad_image_symmetric(Im[:,:,:,0],pad_size,'symmetric')
    return Im_pad


def label_img_pad(in_path,file_ind,BP_path,TP_path,pad_size):
    onlyfiles = ultis.filename_folder(in_path)
    img_filename = onlyfiles[file_ind]
    
    out_name = img_filename.replace('img', 'heatmap')
    BP = ultis.heatmap_mat(BP_path+out_name)
    TP = ultis.heatmap_mat(TP_path+out_name)
    label_img = np.concatenate((BP, TP), axis=-1)
    label_img_pad = ultis.pad_label_symmetric(label_img,pad_size,'symmetric')
    return label_img_pad

def label_points_pad(in_path,file_ind,BP_path,TP_path,pad_size):
    onlyfiles = ultis.filename_folder(in_path)
    img_filename = onlyfiles[file_ind]
    
    out_name = img_filename.replace('img', 'heatmap')
    BP = ultis.heatmap_mat(BP_path+out_name)
    TP = ultis.heatmap_mat(TP_path+out_name)
    label_img = np.concatenate((BP, TP), axis=-1)
    label_points_pad = ultis.pad_label_symmetric(label_img,pad_size,'constant')
    return label_points_pad


#################################################################################################################################
############################# generate one single sample from one pair of original&label image stakcs ###########################


##### for a certain Im_pad,label_img_pad,label_points_pad, a random cropped image is as follow:
##### get the random point and random crop_ID
def crop_ID(label_points_pad,input_size):
    points_ID_array = ultis.array_where((label_points_pad) > 1900)[:,0:3]
    crop_ID = [0,0,0]

    point_ind = np.random.randint(len(points_ID_array))
    points_ID = points_ID_array[point_ind,:]
    crop_ID = points_ID - ultis.random_crop_id(points_ID,input_size)
    
    return crop_ID

##### crop input and output image from the crop_ID
def generate_one_sample(crop_ID,Im_pad,label_img_pad,weights_pad,input_size):
    [x,y,z] = crop_ID
    input_img = Im_pad[x:x+input_size[0],y:y+input_size[1],z:z+input_size[2],:]
    output_img = label_img_pad[x:x+input_size[0],y:y+input_size[1],z:z+input_size[2],:]
    weights_img = weights_pad[x:x+input_size[0],y:y+input_size[1],z:z+input_size[2],:]
    return input_img,output_img,weights_img


def generate_random_crop_ID(points_ID_array,point_ind,input_size):
    crop_ID = points_ID_array[point_ind,:]
    input_size_center = np.array(copy.deepcopy(input_size))
    input_size_center[0] =  int(input_size_center[0]/(math.sqrt(2)))  
    input_size_center[1] =  int(input_size_center[1]/(math.sqrt(2))) 
    input_size_center[2] = input_size_center[2] 
    crop_ID[:,4:7] = crop_ID[:,4:7] - ultis.random_crop_id(crop_ID[:,4:7],input_size_center)
    # print("new",crop_ID[:,4:7])
    return crop_ID
#     ind = np.where(np.logical_and(crop_ID[:,4]<544,crop_ID[:,5]<544,crop_ID[:,6]<93))
#     ind2 = np.where(np.logical_and(crop_ID[:,4]>0,crop_ID[:,5]>0,crop_ID[:,6]>0))
#     ind3 = np.intersect1d(ind,ind2)
#     return crop_ID[ind3]


#     def generate_random_crop_ID(points_ID_array,num_crop_ID,input_size):
#         point_ind = np.random.randint(len(points_ID_array),size = num_crop_ID)
#         points_ID = points_ID_array[point_ind,:]
#         crop_ID = points_ID - ultis.random_crop_id(points_ID,input_size)
#         return crop_ID

################################################################################################################################