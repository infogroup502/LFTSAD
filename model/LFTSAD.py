import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.RevIN import RevIN
from tkinter import _flatten
import torch.nn.functional as F
# try:
#     from tkinter import _flatten
# except ImportError:
#     _flatten = lambda l: [item for sublist in l for item in sublist]
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size) # 第二个全连接层


    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU作为激活函数
        #x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        return x




class LFTSAD(nn.Module):
    def __init__(self, win_size, enc_in,patch_seq,seq_size, c_out, d_model=256,patch_size=[3,5,7], channel=55, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(LFTSAD, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size#list
        self.channel = channel
        self.win_size = win_size
        self.patch_seq=patch_seq###list
        self.seq_size=seq_size## int

        mlp_num_input_size = [self.win_size // patchsize - 1 for patchsize in self.patch_size]
        mlp_num_seq_input_size = [self.win_size // patch_seq - self.seq_size for patch_seq in self.patch_seq]

        # Initialize MLP layers
        self.mlp_size = nn.ModuleList(
            MLP(patchsize - 1, d_model, 1) for patchsize in self.patch_size)

        self.mlp_num = nn.ModuleList(
            MLP(input_size, d_model, 1) for input_size in mlp_num_input_size)

        self.mlp_size_seq = nn.ModuleList(
            MLP((patch_seq - 1) * self.seq_size, d_model, self.seq_size) for patch_seq in self.patch_seq)

        self.mlp_num_seq = nn.ModuleList(
            MLP(input_size, d_model, self.seq_size) for input_size in mlp_num_seq_input_size)



    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel   128 100 51
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')

        
        ###########点级别
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x #128,100,51

            #预处理size
            result=[]
            x_patch_size= rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_size = rearrange(x_patch_size, 'b m (p n) -> (b m) p n', p=patchsize)  # 6258 5 20
            all_indices = list(range(patchsize))
            for i in range(patchsize):  ###排除重构点
                indices = [idx for idx in all_indices if idx != i]
                temp1=x_patch_size[:,indices,:].permute(0,2,1)
                result.append(temp1)


            x_patch_size = torch.cat(result, axis=1).permute(1,0,2)  # 8*105 1 35
            x_patch_size = self.mlp_size[patch_index](x_patch_size).squeeze(-1).permute(1,0).reshape(-1,self.channel,self.win_size).permute(0,2,1)

            num = self.win_size // patchsize
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize)  # 6258 5 20
            result = []
            for i in range (L):
                part = torch.cat((x_patch_num[:,i%patchsize,0:i//patchsize],x_patch_num[:,i%patchsize,i//patchsize+1:num]),dim=1)
                result.append(part)
            x_patch_num = torch.cat(result, axis=0)
            x_patch_num = self.mlp_num[patch_index](x_patch_num)
            x_patch_num = x_patch_num.reshape(B,M,L).permute(0,2,1)#B L M

            series_patch_mean.append(x_patch_size), prior_patch_mean.append(x_patch_num)

        series_patch_mean = list(_flatten(series_patch_mean)) #3
        prior_patch_mean = list(_flatten(prior_patch_mean)) #3

        series_patch_seq = []
        prior_patch_seq = []
        ###########子序列
        for patch_index, patchsize in enumerate(self.patch_seq):
            x_patch_size, x_patch_num = x, x  # 128,100,51

            # 预处理size
            result = []
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_size = rearrange(x_patch_size, 'b m (p n s) -> (b m) p n s', p=patchsize,s=self.seq_size)  # 6258 5 20
            all_indices = list(range(patchsize))
            for i in range(patchsize):
                indices = [idx for idx in all_indices if idx != i]
                temp1 =rearrange( x_patch_size[:, indices, :,:].permute(0,2,1,3),'a b c d -> a b (c d) ')
                result.append(temp1)

            x_patch_size = torch.cat(result, axis=1)  # 8*105 1 35
            x_patch_size = rearrange( self.mlp_size_seq[patch_index](x_patch_size),'(a b) c d  -> a b (c d)',
                                      b=self.channel).permute(0,2,1)

            num = self.win_size // patchsize
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size    128 51 100
            x_patch_num = rearrange(x_patch_num, 'b m (p n s) -> (b m) p n  s', p=patchsize,
                                    s=self.seq_size)  # 6258 5 20
            result = []
            all_indices = list(range(x_patch_num.shape[2]))
            for i in range(x_patch_num.shape[2]):
                indices = [idx for idx in all_indices if idx != i]
                temp1 = rearrange(x_patch_num[:, :, indices, :], 'a b c d ->a b (c d) ')
                result.append(temp1)
            x_patch_num = torch.cat(result, axis=1)
            x_patch_num = self.mlp_num_seq[patch_index](x_patch_num)
            x_patch_num = rearrange(rearrange(x_patch_num,'(a b)  (c  d) e  -> a  b  c  d  e', b=self.channel, d=self.patch_seq[0]).permute(0,1,3,2,4),
                                    ' a b c d e -> a b (c d e)').permute(0,2,1)

            series_patch_seq.append(x_patch_size), prior_patch_seq.append(x_patch_num)

        series_patch_seq = list(_flatten(series_patch_seq))  # 3
        prior_patch_seq = list(_flatten(prior_patch_seq))  # 3

            
        if self.output_attention:
            return series_patch_mean, prior_patch_mean,series_patch_seq, prior_patch_seq
        else:
            return None
        

