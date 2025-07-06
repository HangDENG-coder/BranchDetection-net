import os

device = 'cuda'


###############################################################
# data_generator parameters
###############################################################
# input_size = (64,64,1,8)
# output_size = (56,56,2,6)

input_size = (32,32,1,8)
output_size = (28,28,2,6)
# output_size = (32,32,2,8)

pad_size = input_size
num_batch = 5 ##20 for the best training parameters so far; 5 is for the rotate augmentation
num_epoch = 2000


###############################################################
# interpolate parameters
###############################################################
# num = 8
# input_size = (32,32,1,8)
# output_size = (28,28,2,6)
# enc_chs=(input_size[-2], 1*num, 2*num, 2*num, 4*num, 8*num, 8*num, 16*num, 8*num) 
# enc_ker = [(1,5,5),(1,3,3),(1,1,1)] 
# dec_chs=(enc_chs[-1], 16*num, 8*num, 16*num, 16*num, 2*num, 10*num, 1*num, 2*num)  ## interpolate
# dec_ker =  [(1,1,1),(1,1,1),(1,1,1)] 
# cb_chs =(4*num,2,2) 
# cb_ker = (1,1,1)
# out_sz = output_size[0:2]
# transform_aug = False
###############################################################
# uptranspose parameters
###############################################################
num = 64
# input_size = (64,64,1,8)
# output_size = (56,56,2,6)


input_size = (32,32,1,8)
# output_size = (32,32,2,8)
output_size = (28,28,2,8)


enc_chs=(input_size[-2], 1*num, 2*num, 2*num, 4*num, 8*num, 8*num, 16*num, 32*num) 
enc_ker = [(1,5,5),(1,3,3),(1,1,1)] 

# dec_chs=(enc_chs[-1], 16*num, 8*num, 48*num, 4*num, 2*num, 12*num, 1*num, 2*num)  ## uptranspose
dec_chs=(enc_chs[-1], 8*num, 4*num, 40*num, 4*num, 2*num, 12*num, 1*num, 2*num) 
dec_ker =  [(1,1,1),(1,1,1),(1,1,1)] 
cb_chs =(6*num,12,1)  ## uptranspose
# cb_chs =(6*num,12,2) 
# cb_ker = [(1,3,3),(1,1,1)]
cb_ker = (1,1,1)
out_sz = output_size[0:2]
transform_aug = False


###############################################################
####### the path of the original image ############## 
in_path = '/work/venkatachalamlab/Hang/Transformer/Original_image/Input/'
####### the path of the branch point image ############## 
BP_path = '/work/venkatachalamlab/Hang/Transformer/Original_image/Branch/'
####### the path of the terminal point image ############## 
TP_path = '/work/venkatachalamlab/Hang/Transformer/Original_image/Terminal/'
####### the path of the manual trace image ############## 
trace_path = '/work/venkatachalamlab/Hang/Transformer/Dataset/Label_True/'
###############################################################

###############################################################




###########################################################################################
#           Generate the data in the path
###########################################################################################
points_path = '/work/venkatachalamlab/Hang/Transformer/Dataset/label_points_pad_aug/'
Im_path =  '/work/venkatachalamlab/Hang/Transformer/Dataset/Im_pad_aug/'
label_path =  '/work/venkatachalamlab/Hang/Transformer/Dataset/label_img_pad_aug/'
weights_path =  '/work/venkatachalamlab/Hang/Transformer/Dataset/Weights_img_pad_aug/'
##########################################################################################




























