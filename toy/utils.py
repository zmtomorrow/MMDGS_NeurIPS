import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
palette = sns.color_palette()

from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    Normal,
)



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

def compare(X,Y,p1, p2, ax):
    p1=np.array(p1.detach().cpu())
    p2=np.array(p2.detach().cpu())
    # Removing get_plt() since we're passing the axis directly
    ax.pcolormesh(X, Y, p1.reshape(X.shape[0], Y.shape[0]), alpha=1.0, cmap=sns.light_palette(palette[1], as_cmap=True))
    alphas = np.ones([X.shape[0], Y.shape[0]]) * 0.6
    ax.pcolormesh(X, Y, p2.reshape(X.shape[0], Y.shape[0]), alpha=alphas, cmap=sns.light_palette(palette[2], as_cmap=True))


def MoG4(scale=1.0,std=0.2):
    mean = torch.tensor([[-scale, -scale,], [-scale, scale], [scale, scale], [scale, -scale]])
    comp = Independent(Normal(mean, torch.ones(4,2)*std), 1)
    mix = Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    return MixtureSameFamily(mix, comp)



