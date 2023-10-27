import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from dataclasses import dataclass
from typing import Union
import scipy.special
from numpy import ndarray
from numpy.random import RandomState
from torchvision.utils import make_grid

from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    Normal,
)


import torchvision
from torch.utils import data
from torchvision import transforms


def get_mesh(x_min=-2.,x_max=2.,y_min=-2.,y_max=2.,grid=0.01):
    x = np.arange(x_min, x_max, grid)
    y = np.arange(y_min, y_max, grid)
    X,Y=np.meshgrid(x,y)
    return X,Y

def get_plt():
    plt.figure(figsize=(3,3),dpi=150)
    plt.xlim(-2.0,2.0)
    plt.ylim(-2.0,2.0)
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])

def get_font(color):
        font = {'family': 'serif',
        'color':  color,
        'weight': 'normal',
        'size': 12,
        }
        return font


def mesh_to_density(X,Y,pdf):
    data=np.concatenate((X.reshape(-1,1),Y.reshape(-1,1)),1)
    return pdf(torch.tensor(data,dtype=torch.float32))


def MoG4(scale=1.0,std=0.2):
    mean = torch.tensor([[-scale, -scale,], [-scale, scale], [scale, scale], [scale, -scale]])
    comp = Independent(Normal(mean, torch.ones(4,2)*std), 1)
    mix = Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    return MixtureSameFamily(mix, comp)




def grey_show_many(image,number_sqrt):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()



def grey_save_many(image,number_sqrt,name):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig(name)
    plt.show()


def grey_list_save(image_list,name):
    fig, axs = plt.subplots(1,len(image_list), figsize=(15, 3))
    for ind,img in enumerate(image_list):
        axs[ind].set_axis_off()
        axs[ind].imshow(img, origin="upper", cmap="gray")
    fig.tight_layout()
    fig.savefig(name)
    return None


def grey_return_many(image,number_sqrt):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    return canvas_recon
    # plt.imshow(canvas_recon, origin="upper", cmap="gray")
    # plt.savefig(name)
    # plt.show()


def color_grid(images,sqrt_num=5,save_path=None):
    grid_image=make_grid(images,sqrt_num,pad_value=255)
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(sqrt_num, sqrt_num))
    plt.axis('off')
    if save_path==None:
        plt.imshow(grid_image.permute(1,2,0))
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()
    
    

def color_show_many(image,number_sqrt,dim=32, channels=3):
    image=image.view(-1,3,dim,dim).permute(0,2,3,1)
    canvas_recon = np.empty((dim * number_sqrt, dim * number_sqrt, channels))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim,:] = \
            image[count]
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon)
    plt.show()

def color_return_many(image,number_sqrt,dim=32, channels=3):
    image=image.view(-1,3,dim,dim).permute(0,2,3,1)
    canvas_recon = np.empty((dim * number_sqrt, dim * number_sqrt, channels))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim,:] = \
            image[count]
            count+=1
    return canvas_recon

def color_list_save(image_list,name):
    fig, axs = plt.subplots(1,len(image_list), figsize=(3*len(image_list), 3))
    for ind,img in enumerate(image_list):
        axs[ind].set_axis_off()
        axs[ind].imshow(img)
    fig.tight_layout()
    fig.savefig(name)
    return None

def color_list_show(image_list):
    fig, axs = plt.subplots(1,len(image_list), figsize=(3*len(image_list), 3))
    for ind,img in enumerate(image_list):
        axs[ind].set_axis_off()
        axs[ind].imshow(img)
    fig.tight_layout()
    plt.show()
    return None

def LoadData(opt):
    if opt['data_set'] == 'SVHN':
        train_data=torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,transform=torchvision.transforms.ToTensor())
        
    elif opt['data_set'] == 'CIFAR':
        # if opt['data_aug']==True:
        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()])
        # else:
        #     transform=torchvision.transforms.ToTensor()
        train_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False,transform=transform)
        test_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())

    elif opt['data_set'] == 'CelebA':
        transform=transforms.Compose([
                transforms.CenterCrop(140),
                transforms.Resize(64),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()])
        train_data=torchvision.datasets.CelebA(opt['dataset_path'], split='train',transform=transform)
        test_data=torchvision.datasets.CelebA(opt['dataset_path'], split='test',transform=transform)


    elif opt['data_set']=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='BinaryMNIST':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: torch.round(x),
        ])
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=trans)
    
    else:
        raise NotImplementedError

    train_data_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=opt['test_batch_size'], shuffle=False)
    train_data_evaluation = data.DataLoader(train_data, batch_size=opt['test_batch_size'], shuffle=False)
    return train_data_loader,test_data_loader,train_data_evaluation




dequantize = lambda x: ((x * 255) + torch.rand_like(x)) / 256.0
quantize = lambda x: torch.clamp(torch.floor(x * 256.0) / 255.0, 0, 1)
rescaling = lambda x: 2*x-1
rescaling_inv = lambda x:(x+1)/2


def get_timestamp():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%m%d:%H%M%S")



