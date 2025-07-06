import numpy as np
import sys
import ultis
import load_image
import parameters
import os
import torch
import math
import random
import pickle
import imutils
import scipy
import random
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def instant_rotatation(Im,angle):
    if angle !=0:
        Im_tensor = torch.from_numpy(Im)
        Im_tensor_rot =  TF.rotate(Im_tensor.permute(2,3,0,1),angle).permute(2,3,0,1)
        Im_tensor_rot_array = Im_tensor_rot.numpy()
        return Im_tensor_rot_array
    else:
        return Im

def load_path_rotate(path,stack_ind_aug,angle):
    [file_ind,i,j,k] = stack_ind_aug
    Im = load_arr_pkl(path + str(file_ind)+ str(i)+ str(j)+ str(k)+'.pkl')[0,0,0,0,:,:,:,:]
    return instant_rotatation(Im,angle)

def Rotate_any_degree(img,angle):
#### this function rotate image(x,y,z,channel) for angle degrees
    angle = angle % 365
    img_rotated = np.zeros(img.shape)
    for j in range(img.shape[3]):
        for i in range(img.shape[2]):
            img_rotated[:,:,i,j] = imutils.rotate(img[:,:,i,j], angle=angle)
    return  img_rotated

def Zoom_in_out(img,zoom_factor):
#### this function zoom image(x,y,z,channel) for zoom_factor
    if zoom_factor!=1:
        img_shape = img.shape
        img_zoomed = []
        for j in range(img.shape[3]):
            img_zoomed.append(scipy.ndimage.zoom(img[:,:,:,j], (zoom_factor,zoom_factor,1),order = 0))
        img_zoomed = np.array(img_zoomed).transpose(1,2,3,0)
        
        return  img_zoomed
    else: 
        return img



def inbound_index(points_ID_array,size):
    ind0 = np.logical_and(points_ID_array[:,4]<size[0],points_ID_array[:,5]<size[1])
    ind2 = points_ID_array[:,6]<size[2]
    ind = np.logical_and(ind0,ind2)
    
    ind00 = np.logical_and(points_ID_array[:,4]> 46,points_ID_array[:,5]> 46)
    ind22 = points_ID_array[:,6]> 1
    ind2 = np.logical_and(ind00,ind22)

    return np.logical_and(ind,ind2)


# def get_new_intensity_img(x_original):
#     omega = random.choice([0.5, 1, 2])
#     x_new = abs(torch.sin(omega*math.pi*x_original)) + 0.01 
#     ind = x_new > 1
#     x_new[ind] = 1
#     # x_new = x_original + np.random.normal(0, 0.1, size = (x_original.shape))
#     return x_new

def get_new_intensity_img(x_original):
    omega = random.choice([0.25, 0.5, 0.75, 1])
    x_original[x_original>0.7] = 0.3
    x_new = abs(np.sin(omega*math.pi*x_original)) + 0.01 

    x_new = x_new + np.random.normal(0, 0.1, size = (x_original.shape))
    x_new = x_new/np.max(x_new)

    
    return x_new
    
def load_arr_pkl(filename):
    with open(filename,'rb') as f:
        arr = pickle.load(f)
    f.close()
    return arr

def GaussianBlur_label(img,g):
    for i in range(img.shape[2]):
        for j in range(img.shape[3]):
            img[:,:,i,j] = cv2.GaussianBlur(img[:,:,i,j], (g, g), 0)
            
    # img[img<1e-1] = 0
    return img


def dilate_img(img):
    '''
    img is array with max values = 255
    '''

    gray = img.astype(int)
    gray = gray.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate1 = cv2.dilate(gray, kernel, iterations=1)
    ind = dilate1 > 50

    gray_filtered = img  * ind

    return gray_filtered

def dilate_img_array(img):

    gray = np.max(img[:,:,:,0],axis = 2)
    dilate_gray = dilate_img(gray)[:,:,None]
    return (dilate_gray[:,:,None,:]>0) * img






#####################################################################################
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

def gammaCorrection_3D(src, gamma):
    src = (src).astype(np.uint8)
    ## the input src must be unit 8 with maxmimum value 255 as intensity
    gammaCorrection_3D = np.zeros(src.shape)
    for i in range(src.shape[2]):
        gammaCorrection_3D[:,:,i,0] = gammaCorrection(src[:,:,i,0], gamma)
    return gammaCorrection_3D 

def multiply_gaussian_3D(x_original,omega):
    x_new = x_original * np.random.normal(0,omega, size = (x_original.shape))
    x_new = x_new/np.max(x_new)
    return x_new

def add_gaussian_3D(x_original,omega):
    x_new = x_original/255 + np.random.normal(0,omega, size = (x_original.shape))

    return x_new

def intensity_augmentation(img):
    # img_test = random.choice(
    # [
    # gammaCorrection_3D(img, random.uniform(2.2, 4)), 
    # img,
    # add_gaussian_3D(img, random.uniform(0, 0.1)),
    # img,
    # # multiply_gaussian_3D(img, random.uniform(0.1, 1)), ## cuase broken 
    # # img,
    # gammaCorrection_3D(add_gaussian_3D(img, random.uniform(0.001, 0.002)), random.uniform(1, 2)), ## gamma in range of (3,10) causing overblurred
    # img,
    # add_gaussian_3D(gammaCorrection_3D(img, random.uniform(2, 3)), 1),
    # img,
    # ]
    # )
    # img_test = img_test/255
    if random.uniform(0, 1) < 0.5:
        idx = np.random.randint(0, 3)
        # print("random",idx)
        if idx == 0:
            img_test = gammaCorrection_3D(img, random.uniform(2.2, 4))
        elif idx == 1:
            img_test = add_gaussian_3D(img, random.uniform(0, 0.1))*255
        elif idx == 2:
            img_test = gammaCorrection_3D(add_gaussian_3D(img, random.uniform(0.001, 0.002))*255, random.uniform(1,2))
        elif idx == 3:
            img_test = add_gaussian_3D(gammaCorrection_3D(img, random.uniform(2, 3)), 1)*255
    else:
        img_test = img
    
    return img_test/255

#####################################################################################
class DataGenerator:
    'Generates data for DataGenerator'
    def __init__(self, Im_path, label_path, points_path, weights_path, stack_ind_aug, batch_size, dim_in, dim_out, num_epochs, device,int_aug,rot_aug,zoom_aug,shuffle):

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_epochs = num_epochs
        self.device = device
        self.int_aug = int_aug
        self.rot_aug = rot_aug
        
        self.Im_pad_aug = []
        self.label_img_pad_aug = []
        
        self.Im_path = Im_path
        self.label_path = label_path
        self.points_path = points_path
        self.weights_path = weights_path
        
        self.batch_size = batch_size
        
        
        self.shuffle = shuffle
        self.n = 0
        
        self.edge = (np.array(dim_in) - np.array(dim_out))[0:3]//2
        self.dim_y = tuple(np.array(dim_in)+np.array((0,0,0,1)))
        

        if zoom_aug and random.uniform(0,1) < 0.5:
            self.zoom_factor = random.uniform(0.5, 1)
        else:
            self.zoom_factor = 1
        
        # angle = np.random.randint(365)
        angle = 0
        [file_ind,i,j,k] = stack_ind_aug
        self.label_img_pad_aug = load_path_rotate(self.label_path,stack_ind_aug,angle)
        
        #### This is to blur the label ###
        # self.label_img_pad_aug = GaussianBlur_label(self.label_img_pad_aug,5)   
        
        self.Im_pad_aug = load_path_rotate(self.Im_path,stack_ind_aug,angle)
        # self.Im_pad_aug = intensity_augmentation(self.Im_pad_aug)

        self.Weights_pad_aug = load_path_rotate(self.weights_path,stack_ind_aug,angle)
        

        label_points_pad_aug = load_path_rotate(self.points_path,stack_ind_aug,angle)
        #######  Here is a test only train on BP ##############
        # label_points_pad_aug[:,:,:,1] = 0
        label_points_pad_aug = np.expand_dims(label_points_pad_aug, axis=(0,1,2,3))
        
        
        self.points_ID_array = self.get_points_ID_array(label_points_pad_aug,stack_ind_aug,self.zoom_factor)       

        
        
        if  self.zoom_factor!= 1:
            self.Im_pad_aug = Zoom_in_out(self.Im_pad_aug,self.zoom_factor)
            self.label_img_pad_aug = Zoom_in_out(self.label_img_pad_aug,self.zoom_factor)
            self.Weights_pad_aug = Zoom_in_out(self.Weights_pad_aug,self.zoom_factor)

        
        self.max = self.__len__()


        self.on_epoch_end()
    

    def get_points_ID_array(self,label_points_pad_aug,stack_ind_aug, zoom_factor):
        [file_ind,i,j,k] = stack_ind_aug
        points_ID_array = np.zeros((1,8))
        # num_file_ind = int(len(os.listdir(points_path))/2/2/5)
    
        points_ID = ultis.array_where((label_points_pad_aug) > 1900)
        # ind = inbound_index(points_ID,[544,544,93])
        ind = inbound_index(points_ID,[498,498,93])
        points_ID = points_ID[ind]
        points_ID[:,4:6] = points_ID[:,4:6] * zoom_factor + 0.5
        points_ID[:,0:4] = [file_ind,i,j,k]
        points_ID_array = np.concatenate((points_ID_array,points_ID), axis=0)
        points_ID_array = points_ID_array[1:None]
        
        
        return points_ID_array
    

    def __len__(self):
        return int(np.floor(len(self.points_ID_array)))

        


    def __getitem__(self, index):
    
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Crop_IDs_temp = load_image.generate_random_crop_ID(self.points_ID_array,indexes,self.dim_in)
        
        Crop_IDs_temp = load_image.generate_random_crop_ID(self.points_ID_array,indexes,
                                                           ((self.dim_out[0]),(self.dim_out[1]),self.dim_out[2],1))
        

        
        # Generate data
        X, Y, W = self.__data_generation(Crop_IDs_temp)
        
        if self.device == 'cuda':
            return X.cuda(), Y.cuda(), W
        elif self.device == 'mps':
            return X.to(device = torch.device("mps")), Y.to(device = torch.device("mps")), W.to(device = torch.device("mps"))
        elif self.device == 'cpu':
            return X,Y, W
    

    def on_epoch_end(self):
        

        # self.indexes = np.arange(len(self.points_ID_array))
        self.indexes = np.arange((self.max))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            


            
            
    def __data_generation(self, Crop_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        

        X = torch.zeros((self.batch_size, *self.dim_in))

        Y = torch.zeros((self.batch_size, *self.dim_y))

        W = torch.zeros((self.batch_size, *self.dim_in))

        # Generate data for one batch
    
    

        for idx, ID in enumerate(Crop_IDs_temp):


            # Store sample images                    
            [file_ind,i,j,k] = ID[0:4].astype(int)
            
            crop_ID = ID[4:7].astype(int)

            
          
            x,y,w = load_image.generate_one_sample(crop_ID, self.Im_pad_aug, self.label_img_pad_aug, self.Weights_pad_aug, self.dim_in)


            # x = intensity_augmentation(x)
            
   
            # x = dilate_img_array(x)
            

            
            if self.rot_aug == True:
                angle = np.random.randint(365)               
                y_test = Rotate_any_degree(y,angle)
                while np.max(y_test) < np.max(y)-10:
                    angle = np.random.randint(365)               
                    y_test = Rotate_any_degree(y,angle)
                    
                y = y_test    
                x = Rotate_any_degree(x,angle)
                w = Rotate_any_degree(w,angle)
                
            
            if self.int_aug == True:
                # X[idx] = get_new_intensity_img(torch.from_numpy(x))
                
                X[idx] = torch.from_numpy(get_new_intensity_img(x))
            else:
                X[idx] = torch.from_numpy(x)

            W[idx] = torch.from_numpy(w)
            
            # y = y/np.max(y)
            # y = y * (x>0)
            # y[y>0] = 1
            Y[idx] = torch.from_numpy(y)
            
    
        X = X.permute(0,4,3,2,1)
        Y = Y.permute(0,4,3,2,1)
        W = W.permute(0,4,3,2,1)
        
        
#         X = torch.cat((X,X),1)

        # Y = Y[:,self.edge[0]: -self.edge[0],self.edge[1]:-self.edge[1],self.edge[2]:-self.edge[2],:].permute(0,4,3,2,1)
        
        # Y = Y[:,self.edge[0]: -self.edge[0],self.edge[1]:-self.edge[1],1:-1,:].permute(0,4,3,2,1)
        
        
        ##########################################
        # ### works with the U_classify###
        # ind = (Y[:,0] + Y[:,1] )> 0
        # Y = torch.cat((ind[:,None],Y),dim = 1)
        # Y = torch.argmax(Y, dim = 1)
        ##########################################
        # return X[:,:,:,7:-7,7:-7],Y[:,:,:,7:-7,7:-7]
        return X,Y[:,0:1,:], W

        # Y_f = torch.cat([W>0 -Y[:,0:1,:]>0,Y[:,0:1,:],], dim=1)
        # return X, Y_f, W_f




    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.n >= (self.max/self.batch_size):            
            self.on_epoch_end()
            self.n = 0
            raise StopIteration

        else:
            result = self.__getitem__(self.n)
            self.n += 1

            return result