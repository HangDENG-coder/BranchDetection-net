import torch
import torch.nn as nn
import torchvision
import math
from typing import Optional, List
# from labml import tracker
import parameters
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

################################################################################################   
#                         Augmentaiton the datasets in rotation/flip/reverse
################################################################################################     

class DeterministicTransform:
    """Rotate by the given angle."""

    def __init__(self, angle = 0, h_flip = False,v_flip = False,reverse = False):
        self.angle = angle
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.reverse = reverse

    def __call__(self, x):
        if self.angle!=0:
            x = torch.rot90(x, self.angle , dims=[3,4])
           
        if self.h_flip:
            x = TF.hflip(x) #horizontal flip
            
        if self.v_flip:
            x = TF.vflip(x) #horizontal flip
            
        if self.reverse:
            x = torch.flip(x, dims=(2,))
        return x

################################################################################################   
#                         Build the MultiHeadAttentionBlock
################################################################################################     
    
class MultiHeadAttentionBlock(nn.Module):
    """
    ### Attention block
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """


    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 1):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
                
        
        # Default `d_k`
        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
   
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)

        self.output = nn.Linear(n_heads * d_k, n_channels)
        
        self.dropout = nn.Dropout()

        self.scale = 1 / math.sqrt(d_k)

        self.n_heads = n_heads

        self.d_k = d_k
        
    def get_mask(self, mask: torch.Tensor, q_shape: List[int], k_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """

        assert mask.shape[0] == 1 or mask.shape[0] == q_shape[0]
        assert mask.shape[1] == k_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == q_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)
        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    
    
    def get_QKV(self,x:torch.Tensor):
        batch_size = x.shape[0]
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        return q,k,v
    
    
    def get_score(self,x:torch.Tensor,scale):
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        q,k,v = self.get_QKV(x)    
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        score = torch.einsum('bijh,bjhd->bihd', attn, v)
        score = self.dropout(score)
        return score,[q.shape,k.shape,v.shape]
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, n_channels, depth, height, width]`
        """

        batch_size, n_channels, deepth, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.contiguous().view(batch_size, n_channels, -1).permute(0, 2, 1)
         
        
        score,[q_shape,k_shape,v_shape] = self.get_score(x,self.scale)
        
        
        if mask is not None:
            mask = self.get_mask(mask, q_shape, k_shape)
            score = rscore.masked_fill(mask == 0, float('-inf'))
        
        # Reshape to `[batch_size, seq, n_heads * d_k]`

        score = score.contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # Transform to `[batch_size, seq, n_channels]`
        score = self.output(score)
             
        # Add skip connection
        score += x

        # Change to shape `[batch_size, in_channels,deepth, height, width]`
        score = score.permute(0, 2, 1).view(batch_size, n_channels, deepth, height, width)


        return score
 
    
################################################################################################   
#                         Build the ConvolutionBlock
################################################################################################ 
class ConvolutionBlock(torch.nn.Module):

    def __init__(self,chs,kernel_size):
        super().__init__()
        # print("kernel_size",kernel_size)
        # print(kernel_size[2])
        pad = (0,0,0)
        if len(kernel_size)==3:
            if kernel_size[2] == 3:
                pad = (0,1,1)    
            elif kernel_size[2] == 5:
                pad = (0,3,3)
            elif kernel_size[2] == 7:
                pad = (0,3,3)
  
            
        self.conv1 = torch.nn.Conv3d(chs[0], chs[1], kernel_size, stride = 1,padding = pad, padding_mode = 'reflect')
        self.norm1 = nn.InstanceNorm3d(chs[1]*2)   ### expected channel
        self.activation = torch.nn.ReLU()
        
        if len(kernel_size)==3:
            if kernel_size[2] == 3:
                pad = (0,1,1)
            elif kernel_size[2] == 5:
                pad = (0,1,1)
            elif kernel_size[2] == 7:
                pad = (0,3,3)
        self.conv2 = nn.Conv3d(chs[1], chs[2], kernel_size, stride = 1,padding = pad, padding_mode = 'reflect')
        self.norm2 = nn.InstanceNorm3d(chs[2]*2)
        
    def forward(self, x):
        x = self.conv1(x) 
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x
    
    
class EncoderBlock(nn.Module):
    def __init__(self, chs,kernels):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvolutionBlock(chs[i:i+3], kernels[int((i+1)/3)]) for i in range(0,len(chs),3)])
        self.pool = nn.MaxPool3d(2, stride = 2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)  
            ftrs.append(x)
            x = self.pool(x)

        return ftrs
    
    
    
class UpConvSampling(nn.Module):
    def __init__(self, chs):   
        super().__init__()
        self.conv1 = torch.nn.Conv3d(chs, chs, (1,1,1), stride = 1)
        self.conv2 = torch.nn.Conv3d(int(chs*2), int(chs*2), (1,1,1), stride = 1)
        self.norm1 = nn.InstanceNorm3d(chs*2)   ### expected channel
        self.activ = torch.nn.ReLU()
        self.upsample = nn.Upsample(scale_factor = (2,2,2), mode='nearest')
        # print(chs)
        
    def forward(self, x):
        # print("before",x.shape)
        x = nn.functional.interpolate(x,scale_factor = (2,2,2), mode = 'nearest')
        # print("after",x.shape)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ(x)
        return x    
    
    
    
    
class DecoderBlock(nn.Module):
    def __init__(self, chs,kernels):
        super().__init__()
        self.dec_blocks = nn.ModuleList([ConvolutionBlock(chs[i:i+3], kernels[int((i+1)/3)]) for i in range(0,len(chs),3)])
        self.upconvs    = nn.ModuleList([nn.ConvTranspose3d(chs[i+2], 2*chs[i+2], 2, 2) for i in range(0,len(chs),3)])
        # self.upconvs    = nn.ModuleList([UpConvSampling(chs[i+2]) for i in range(0,len(chs),3)])
    
    # def crop(self, enc_ftrs, x):
    #     _, _, H, W = x.shape
    #     enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
    #     return enc_ftrs  

    
    def crop(self, enc_ftrs, x):
        _, _, D, H, W = x.shape
        d0 = (enc_ftrs.shape[2] - D) // 2
        h0 = (enc_ftrs.shape[3] - H) // 2
        w0 = (enc_ftrs.shape[4] - W) // 2
        
        enc_crop = enc_ftrs[:, :, d0:d0+D, h0:h0+H, w0:w0+W]
        return enc_crop
    
    def forward(self, x, encoder_features):
        for i in range(len(encoder_features)):
            x        = self.dec_blocks[i](x)
            x        = self.upconvs[i](x)
                                           
            enc_ftrs = encoder_features[i]
            # print("before cat",x.shape,enc_ftrs.shape)
            enc_ftrs = self.crop(enc_ftrs, x)
            # print("after cat",x.shape,enc_ftrs.shape)
            x        = torch.cat([x, enc_ftrs], dim=1)
            # print("after cat",x.shape)

        return x
    
def compute_the_larger_points(y_tensor):
    ind = torch.argmax(y_tensor,dim = 1,keepdim = True)
    return y_tensor * torch.cat((1 - ind, ind), 1) 


    
    
class UNet(nn.Module):
    def __init__(self, 
                 enc_chs,    enc_ker,
                 dec_chs,    dec_ker,
                 cb_chs,      cb_ker,
                 out_sz,
                 transform_aug = False):
        super().__init__()
        
        self.transform_aug = transform_aug
        if self.transform_aug:
            angle = np.random.randint(4)
            [h_flip,v_flip,reverse] = np.array(np.random.randint(2,size = 3), dtype=bool)
            self.transform = DeterministicTransform(angle = angle,h_flip = h_flip,v_flip = v_flip,reverse = reverse)
            self.reverse_transform =  DeterministicTransform(angle = int(4-angle),h_flip = h_flip,v_flip = v_flip,reverse = reverse)
            
        self.encoder     = EncoderBlock(enc_chs, enc_ker)
        self.decoder     = DecoderBlock(dec_chs, dec_ker) 
        self.convblk     = ConvolutionBlock(cb_chs,cb_ker)
        self.pool        = nn.MaxPool3d(2, stride = 2)
        self.out_sz      = out_sz
        
        self.conv = torch.nn.Conv3d(cb_chs[0], cb_chs[1], (1,1,1), stride = 1)
        self.conv2 = torch.nn.Conv3d(cb_chs[1], cb_chs[2], (1,1,1), stride = 1)
        self.conv3 = torch.nn.Conv3d(3, 12, (1,1,1), stride = 1)
        self.conv4 = torch.nn.Conv3d(12,3, (1,1,1), stride = 1)
        # self.conv3 = torch.nn.Conv3d(2, cb_chs[2], (1,1,1), stride = 1)
        
        self.norm1 = nn.InstanceNorm3d(cb_chs[1]*2) 
        self.norm2 = nn.InstanceNorm3d(cb_chs[2])   ### expected channel
        self.actv1 = torch.nn.ReLU()
        self.actv2 = torch.nn.Softmax(dim=1)
        
        # self.attn        = MultiHeadAttentionBlock(1,1,1,1)
        # self.attn2       = MultiHeadAttentionBlock(96,3,32,1)
        self.attn3       = MultiHeadAttentionBlock(2,2,2,1)

        
        
        
    def forward(self, x):
        # x = self.attn(x)
        
        if self.transform_aug:
            x = self.transform(x)
            
        
        enc_ftrs = self.encoder(x)
        # print("encoder",enc_ftrs[1].shape)
        DB_inputs= self.pool(enc_ftrs[-1])     
        # print("DB_inputs ",DB_inputs.shape)
        
        # DB_inputs = self.attn2(DB_inputs)

        decoder  = self.decoder(DB_inputs, enc_ftrs[::-1])

        # decoder = self.attn(decoder)
        # print("decoder",decoder.shape)
        ###############################################################
        ################### The following is a test ###################
        
        out = self.conv(decoder) 
        out = self.norm1(out)
        out = self.actv1(out) 
        # print("conv1 ", out.shape)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.actv2(out) 
        
        
#         out = self.conv3(out)
#         out = self.norm1(out)
#         out = self.actv1(out) 
        
#         out = self.conv4(out)
#         out = self.norm2(out)
#         out = self.actv1(out) 

        # out = self.conv3(out)
        # out = self.norm2(out)
        # out = self.actv1(out) 

        
        # print("out", out.shape)
        # out = self.attn3(out)
        # out = self.actv2(out) 
        
        # out = out * (x>0)
        ###############################################################
        if self.transform_aug:
            out  = self.reverse_transform(out)
        
        out      = torchvision.transforms.CenterCrop([self.out_sz[0], self.out_sz[1]])(out)     
        # print("out", out.shape)
        # out      = out[:,:,1:-1,:,:]
        
        # out = out / torch.max(out)
        # scale = torch.amax(out,dim = (2,3,4)) + 1e-5
        # out = (out/scale[:,None,None,None])
        # out = out*(x>0)
        # return out[:,1:3,:,:,:]
        return out


    
    
    
################################################################################################   
#                         Build the loss funciton along Branch and Terminal
################################################################################################ 
    


def iou_coef(y_pred, y_true, smooth=1e-0):
    # y_one = nn.functional.one_hot(y_true.to(torch.int64), num_classes=1)[:, :, :, :, :, 0]
    
    y_true = y_true/torch.max(y_true)
    y_pred = y_pred/troch.max(y_pred)
    intersection = torch.sum((y_true * y_pred), axis=[1,2,3,4])
    union = torch.sum(y_true,[1,2,3,4])+torch.sum(y_pred,[1,2,3,4]) - intersection
    return torch.mean((union) / (intersection + smooth), axis=0)

def iou_coef_one_layer(y_pred, y_true, smooth,dim):
    one_layer = iou_coef(y_pred[:,dim:dim+1,:,:,:], y_true[:,dim:dim+1,:,:,:], smooth=smooth)
    return one_layer

def sumed_iou_coef(y_pred, y_true, smooth=1e-0):
    return iou_coef_one_layer(y_true, y_pred, smooth = smooth,dim = 0) + iou_coef_one_layer(y_true, y_pred, smooth = smooth,dim = 1) 

def BP(y_pred, y_true, smooth=1e-0):
    return iou_coef_one_layer(y_true, y_pred, smooth = smooth,dim = 0)

def TP(y_pred, y_true, smooth=1e-0):
    return iou_coef_one_layer(y_true, y_pred, smooth = smooth,dim = 1)

def sumed_iou_coef(y_pred, y_true, smooth=1e-0):
    return iou_coef_one_layer(y_true, y_pred, smooth = smooth,dim = 0) + iou_coef_one_layer(y_true, y_pred, smooth = smooth,dim = 1)    
    
    
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
