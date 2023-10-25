import torch.nn as nn
import torch.nn.functional as F
import torch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class FeedFowardNet(nn.Module):
    def __init__(self,  input_dim=784, output_dim=1, h_dim=400, h_layer_num=1,act='tanh', if_bn= False):
        super().__init__()
        self.idenity=Identity()
        self.input_dim=input_dim
        self.h_layer_num=h_layer_num
        self.fc_list=nn.ModuleList([])
        self.bn_list=nn.ModuleList([])
        if act=='tanh':
            self.act=torch.tanh
        elif act=='relu':
            self.act=torch.relu
        elif act=='swish':
            self.act=lambda x: x*torch.sigmoid(x)
        elif act=='leakyrelu':
            self.act=F.leaky_relu
        
        for i in range(0,h_layer_num+1):
            if i==0:
                self.fc_list.append(nn.Linear(input_dim, h_dim))
            else:
                self.fc_list.append(nn.Linear(h_dim, h_dim))
            if if_bn:
                self.bn_list.append(nn.BatchNorm1d(h_dim))
            else:
                self.bn_list.append(self.idenity)
        self.fc_out = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        x=x.view(-1,self.input_dim)
        for i in range(0,self.h_layer_num+1):
            x=self.act(self.bn_list[i](self.fc_list[i](x)))
        return self.fc_out(x)


class EnergyNet(nn.Module):
    def __init__(self,  input_dim=784, sigma=0.1, h_dim=400, h_layer_num=1,act='swish', if_bn= False):
        super().__init__()
        self.idenity=Identity()
        self.input_dim=input_dim
        self.h_layer_num=h_layer_num
        self.sigma=sigma
        self.fc_list=nn.ModuleList([])
        self.bn_list=nn.ModuleList([])
        if act=='tanh':
            self.act=torch.tanh
        elif act=='relu':
            self.act=torch.relu
        elif act=='swish':
            self.act=lambda x: x*torch.sigmoid(x)
        elif act=='leakyrelu':
            self.act=F.leaky_relu
        
        for i in range(0,h_layer_num+1):
            if i==0:
                self.fc_list.append(nn.Linear(input_dim, h_dim))
            else:
                self.fc_list.append(nn.Linear(h_dim, h_dim))
            if if_bn:
                self.bn_list.append(nn.BatchNorm1d(h_dim))
            else:
                self.bn_list.append(self.idenity)
        self.fc_out = nn.Linear(h_dim,input_dim)

    def forward(self, x):
        x_mid=x.view(-1,self.input_dim)
        for i in range(0,self.h_layer_num+1):
            x_mid=self.act(self.bn_list[i](self.fc_list[i](x_mid)))
        return torch.sum((self.fc_out(x_mid)-x)**2,-1)/(2*(self.sigma**2))






