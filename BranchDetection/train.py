import pandas as pd
import os
import parameters
import copy
import random
import math
import sys
# from U_Classify import *
from U_attention import *
from train_generator import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
####################################################################################################
#   this version works as a test
# trained_model: criteration(yhat,targets)
# trained_model2: criteration(yhat,targets) - criteration(targets,targets)
####################################################################################################

def save_arr_pkl(filename,arr):
    with open(filename,'wb') as f:
        x = pickle.dump(arr,f)
    f.close()

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

model = UNet(parameters.enc_chs,    parameters.enc_ker,
             parameters.dec_chs,    parameters.dec_ker,
             parameters.cb_chs,     parameters.cb_ker,
             parameters.out_sz,     transform_aug = parameters.transform_aug)

if parameters.device == 'cuda':
    model = model.cuda()
elif parameters.device == 'mps':
    model = model.to(torch.device("mps"))
elif parameters.device == 'cpu':
    model = model
    
init_all(model, torch.nn.init.normal_, mean=0., std=1)


num_epoch = 20
val_stacks = [0]
# train_stacks = [0,1,2]
train_stacks = list(set(list(range(16))) - set(val_stacks) - set([6]))
save_folder = './U_intensity_att1/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

    
    
def Blur_prediction(yhat):
    GB = torchvision.transforms.GaussianBlur((5,5), sigma=(0.1, 2.0))
    for i in range(len(yhat)):
        yhat[i] = GB(yhat[i].clone())
    return yhat/torch.max(yhat)

def iou_coef(y_pred, y_true,  smooth):
    if torch.max(y_true)!=0:
        y_true = y_true/torch.max(y_true)
    if torch.max(y_pred)!=0:
        y_pred = y_pred/torch.max(y_pred)

    
    # y_true = (nn.functional.normalize(y_true, p=1.0, dim = (0,1) ))
    # y_pred = (nn.functional.normalize(y_pred, p=1.0, dim = (0,1) ))

    union = torch.sum(y_true,[1,2,3,4]) + torch.sum(y_pred,[1,2,3,4]) 
    intersection = torch.sum((y_true * y_pred), axis=[1,2,3,4])
    # intersection = torch.sum((Blur_prediction(y_true.clone()) * (y_pred)), axis=[1,2,3,4])
    # union = torch.sum(y_pred*(y_true<1e-2),[1,2,3,4])
    # union[intersection==0] = union[intersection==0] * 2
    
 

    coef = union / (intersection + smooth  ) 
    # coef = ( 1 - intersection/(union + 1e-5)) 
    
    return torch.mean(coef, axis=0)


def uoi_coef(y_pred, y_true):
    y_BP = (y_pred).clone() 
    y_TP = (y_true).clone()
    
    # y_BP[y_BP > 0] = 1
    # y_TP[y_TP > 0] = 1
    if torch.max(y_BP)!=0:
        y_BP = y_BP/torch.max(y_BP)
    if torch.max(y_TP)!=0:
        y_TP = y_TP/torch.max(y_TP)
    
    intersection = torch.sum((y_BP * y_TP), axis=[1,2,3,4])
    return torch.sum(intersection, axis=0)
                      

def iou_coef_one_layer(y_pred, y_true, smooth,dim):
    one_layer = iou_coef(y_pred[:,dim:dim+1,:,:,:], y_true[:,dim:dim+1,:,:,:], smooth=smooth)
    return one_layer

def BP(y_pred, y_true, smooth=1e-0):
    return iou_coef_one_layer(y_pred, y_true,  smooth = smooth,dim = 0)

def TP(y_pred, y_true, smooth=1e-0):
    return iou_coef_one_layer(y_pred, y_true, smooth = smooth,dim = 1)

def sumed_iou_coef(y_pred, y_true, smooth=1e-0):
    return BP(y_pred, y_true, smooth = smooth)
    # return BP(y_pred, y_true, smooth = smooth) + TP(y_pred, y_true, smooth = smooth)
    # return BP(y_pred, y_true, smooth = smooth) + TP(y_pred, y_true, smooth = smooth) +  uoi_coef(y_pred[:,0:1,:,:,:], y_pred[:,1:2,:,:,:])
                      

def save_model(save_folder, epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print("Saving final model at epoch "+ str(epochs))
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_folder+criterion +'.pth')



# criterion = sumed_iou_coef(epoch = 0, smooth = 0.1)
optimizer = torch.optim.SGD(list(model.parameters()), lr=1e-2, momentum=0.9)

# if os.path.exists('./trained_model/val_loss.pth'):
#     model.load_state_dict((torch.load('./trained_model/val_loss.pth'))['model_state_dict'])
#     optimizer.load_state_dict((torch.load('./trained_model/val_loss.pth'))['optimizer_state_dict'])

# key = 'loss'
# load_folder = './U_intensity2/'
# model.load_state_dict((torch.load(load_folder + key + '.pth',map_location=torch.device('cpu')))['model_state_dict'])
# torch.load(load_folder + key + '.pth')['epoch']    
    
    
    
# criterition = nn.CrossEntropyLoss()


criterition = nn.NLLLoss()
training_history = {}
for epoch in range(parameters.num_epoch):
    
    if epoch == 0:
        for g in  optimizer.param_groups:
            g['lr'] = 1e-2
        
    elif epoch > (np.argmin(history[key_name]) + 10) and g['lr'] > 1e-5:
        for g in  optimizer.param_groups:
            g['lr'] = 1e-2/10
        
    
    model.train(True)
    [num_batches,loss_epoch,loss_BP,loss_TP,loss_relative] = [0, 0, 0, 0, 0]
    [int_aug, rot_aug, zoom_aug] = np.array(np.random.randint(2, size=3), dtype=bool)

    random.shuffle(train_stacks)
    # train_stacks =[0]
    for stack in (train_stacks):
        train_data = DataGenerator(parameters.Im_path,parameters.label_path,parameters.points_path,parameters.weights_path,  
                            stack_ind_aug = [stack,0,0,0],
                            batch_size = parameters.num_batch, 
                            dim_in = (46,46,8,1), 
                            dim_out = (46,46,8,2), 
                            num_epochs = 1, 
                            device = parameters.device,
                            int_aug = False,
                            rot_aug = True,
                            zoom_aug = False,
                            shuffle= True)
        
        for i, (inputs, targets, weights) in enumerate(train_data):           
            # clear the gradients
            optimizer.zero_grad()
            yhat = model(inputs)
            # loss_BP += BP(yhat, targets)
            # loss_TP += TP(yhat, targets)
            
            loss = sumed_iou_coef(yhat, targets, smooth = 1e-2)
            # loss = criterition(yhat, targets)
            
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

            loss_epoch += loss
            loss_relative += sumed_iou_coef(yhat, targets, smooth = 0.1)
            # loss_relative += loss - criterition(targets, targets,smooth = 0.1)
            if math.isnan(loss.item()):
                print( [stack,i],loss_epoch.item())
                print("NaN")
                
                break
        # print([stack,i],loss_epoch.item()/(num_batches+1))
        num_batches += i
        
    writer.add_scalar("Loss/train_all_change_loss", loss_epoch.item()/(num_batches+1), epoch)
        
    

    training_history.update({epoch:{'loss':loss_epoch.item()/(num_batches+1)}})
    # training_history[epoch].update({'BP':loss_BP.item()/(num_batches+1)})
    # training_history[epoch].update({'TP':loss_TP.item()/(num_batches+1)})
    training_history[epoch].update({'loss_relative':loss_relative.item()/(num_batches+1)})
    
    
    
#     model.train(False)
#     [val_num_batches,val_loss_epoch,val_loss_BP,val_loss_TP] = [0, 0, 0, 0]
#     for stack in (val_stacks):
#         val_data = DataGenerator(parameters.Im_path,parameters.label_path,parameters.points_path, 
#                             stack_ind_aug = [stack,0,0,0],
#                             batch_size = parameters.num_batch, 
#                             dim_in = (32,32,8,1), 
#                             dim_out = parameters.output_size, 
#                             num_epochs = 1, 
#                             device = parameters.device,
#                             int_aug = False,
#                             rot_aug = False,
#                             zoom_aug = False,
#                             shuffle= False)
        
#         for i, (inputs, targets) in enumerate(train_data):           
#             # clear the gradients
#             optimizer.zero_grad()
#             yhat = model(inputs)
#             val_loss = sumed_iou_coef(yhat, targets, smooth = 0.1) - sumed_iou_coef(targets, targets, smooth = 0.1)
        
#             val_loss_epoch += val_loss
#             val_loss_BP += BP(yhat, targets)
#             val_loss_TP += TP(yhat, targets)

#         val_num_batches += i

    
#     training_history[epoch].update({'val_loss':val_loss_epoch.item()/(val_num_batches+1)})
#     training_history[epoch].update({'val_BP':val_loss_BP.item()/(val_num_batches+1)})
#     training_history[epoch].update({'val_TP':val_loss_TP.item()/(val_num_batches+1)})
    
    # print(epoch,training_history[epoch]['loss'],training_history[epoch]['val_loss'],training_history[epoch]['loss_relative'])
    save_arr_pkl(save_folder +"training_history.pkl",training_history)
    print(epoch,training_history[epoch]['loss'],training_history[epoch]['loss_relative'])
    sys.stdout.flush()
    history = pd.DataFrame(pd.read_pickle(save_folder + "training_history.pkl")).T
    # for key_name in ['loss','BP','TP','val_loss','val_BP','val_TP', 'loss_relative']:
    for key_name in ['loss', 'loss_relative']:
        if epoch == np.argmin(history[key_name]):
            save_model(save_folder, epoch, model, optimizer, key_name)


writer.close()            
            
