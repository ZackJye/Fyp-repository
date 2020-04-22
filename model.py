import os
import torch
import torchvision
import numpy as np
from torchvision import transforms, utils
import pandas as pd
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from skimage.io import imread
from skimage import io, transform
from PIL import Image 
data_path = "./jpegs_256/"    # define UCF-101 RGB data path
action_name_path = './UCF101actions.pkl'
save_model_path = "./ResNetCRNN_ckpt/"

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.5       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 5             # number of target category
epochs = 10        # training epochs
batch_size = 40  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1
device = "cuda" if torch.cuda.is_available() else "cpu"

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      
class MyResnet(nn.Module):
    def __init__(self, inp = 2048, h1=1024, out = 5, d=0.30):
        super().__init__()
        resnet = torchvision.models.resnet50()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(inp*2,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(d)
        self.fc1 = nn.Linear(inp*2, h1)
        self.bn1 = nn.BatchNorm1d(h1,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(d)
        self.fc2 = nn.Linear(h1, out)
        for m in self.modules():
          if isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.resnet(x)
        ap = self.ap(x)
        mp = self.mp(x)
        x = torch.cat((ap,mp),dim=1)
        x = self.fla(x)
        x = self.bn0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)         
        x = torch.sigmoid_(self.fc2(x))
        
        return x

class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=5):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        X=[]
        X2=[]
        #a= torch.argmax(RNN_out,dim=1)
        #print(a.shape)
        #print(a)
        #print(RNN_out[:,-1,:])
       
        for t in range(RNN_out.size(1)):
        # a = score_max(RNN_out,1,RNN_out)
          x1=RNN_out[:,t,:]
        #a = a.type(torch.cuda.FloatTensor)
          x1 = self.fc1(x1)   # choose RNN_out at the last time step
          x1 = F.relu(x1)
          x1 = F.dropout(x1, p=self.drop_p, training=self.training)
          x1 = self.fc2(x1)
          x1 = torch.sigmoid(x1)
          X.append(x1)
        # X=numpy.array(X) 
        # x=torch.mean(torch.cuda.FloatTensor(X))
        #print(X)
        #print(torch.stack(X).shape)
        #print(torch.stack(X).shape)
        x3=torch.mean(torch.stack(X),dim=0)
        #print(x3.shape)
        # for i in x3.size(1):
        #   s=x3[:,i,:]
        #    x=torch.mean(torch.stack(X2),dim=1)
        # X2.append(x3)
        # print(torch.stack(X2).shape)
        # x=torch.mean(torch.stack(X2),dim=1)
        

        return x3

class ResCNNEncoder(nn.Module):
    def __init__(self,modelA, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()
        modules=list(modelA.children())[:-9]
        self.model=nn.Sequential(*modules)
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))
        self.fla = Flatten()
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        #modules = list(model_ensemble.children())     # delete the last fc layer.
        #self.model1 = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048*2, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                images=x_3d[:,t, :, :, :]
                images = images.view(-1,3, 224, 224)  
                x = self.model(images.type(torch.cuda.FloatTensor))  
                #x = torch.cat((x1, x2), dim=1)
                #x = x1.view(x1.size(0), -1)             # flatten output of conv

            # FC layers
            ap = self.ap(x)
            mp = self.mp(x)
            x = torch.cat((ap,mp),dim=1)
            x = self.fla(x)
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### START CODE HERE ### (6 lines for linear, 5 lines for batch norm)         
        layer_sizes=[26,256,128]
        layer_sizes1=[256,128]
        self.fc= nn.ModuleList([nn.Linear(layer_sizes[i-1],layer_sizes[i]) for i in range(1,len(layer_sizes))])
        self.bn= nn.ModuleList([nn.BatchNorm1d(layer_sizes1[i]) for i in range(0,len(layer_sizes1))])
        self.dropout=nn.Dropout(0.3)
        ### END CODE HERE ###
        
        
        # Initialize all layers
        ### START CODE HERE ### (4 lines) 
        for m in self.modules():
          if isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
        ### END CODE HERE ###

                        
    def forward(self, x):
        ### START CODE HERE ### 
        x= x.view(x.size(0),-1)
        for i in range(0,len(self.fc)):
          x=torch.relu(self.dropout(self.bn[i](self.fc[i](x))))

        #x= self.fc[-1](x)
        #x=torch.sigmoid(x)

        # (7 to 18 lines - 1 line to flatten input, 6 lines for linear, 5 lines for bn, 6 lines for relu)  
        ### END CODE HERE ###
        
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,modelC):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC=  modelC
        layer_sizes=[133,128,64,5]
        layer_sizes1=[128,64]
        self.fc= nn.ModuleList([nn.Linear(layer_sizes[i-1],layer_sizes[i]) for i in range(1,len(layer_sizes))])
        self.bn= nn.ModuleList([nn.BatchNorm1d(layer_sizes1[i]) for i in range(0,len(layer_sizes1))])
        self.dropout=nn.Dropout(0.3)
        
    def forward(self, x1, x2):
        x1 = self.modelC(self.modelB(x1))
        x2 = self.modelA(x2)
        x = torch.cat((x1, x2), dim=1)
        x= x.view(x.size(0),-1)
        for i in range(0,len(self.fc)-1):
          x=torch.relu(self.dropout(self.bn[i](self.fc[i](x))))

        x= self.fc[-1](x)
        x=torch.sigmoid(x)

        return x

def getModel():
  model=MyResnet()
  cnn_encoder = ResCNNEncoder(model,fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=0.5, CNN_embed_dim=CNN_embed_dim).to(device)
  rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=0.5, num_classes=k).to(device)
  model_audio=Audio()
  model_ensemble=MyEnsemble(modelA=model_audio,modelB=cnn_encoder,modelC=rnn_decoder) 
  return model_ensemble