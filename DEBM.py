import numpy as np
import torch
import argparse
from torch import optim
from tqdm import tqdm
from networks import *
from utils import *
from torch.autograd import grad
import os
import json


opt={
    'dataset_path':'../data/',
    'epochs':1000,
    'test_batch_size':100,
    'seed':0,
    'lr':1e-4,
    'save_path':'./save/',
}

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--data', type=str, default="MNIST")
parser.add_argument('--std', type=float, default="0.3")
parser.add_argument('--learn', type=str, default="energy")
parser.add_argument('--energy', type=str, default="sum")
parser.add_argument('--rescale', type=bool, default=False)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)

opt['device']=parser.parse_args().device 
opt['x_std']=parser.parse_args().std
opt['data_set']=parser.parse_args().data
opt['learn']=parser.parse_args().learn
if opt['learn']=='energy':
    opt['energy']=parser.parse_args().energy
opt['load_name']=parser.parse_args().load
opt['save_freq']=parser.parse_args().save_freq
opt['batch_size']=parser.parse_args().batch_size
opt['rescale']=parser.parse_args().rescale
if parser.parse_args().name==None:
    opt['name']=get_timestamp()
else:
    opt['name']=parser.parse_args().name


opt['save_path']=opt['save_path']+opt['data_set']+'/'+opt['learn']+'/'+opt['name']+'/'
if opt['learn']=='energy':
    opt['save_path']=opt['save_path']+opt['energy']+'_'+str(opt['x_std'])+'/'
elif opt['learn'] in ['score','kl']:
    opt['save_path']=opt['save_path']+str(opt['x_std'])+'/'


os.makedirs(opt['save_path'], exist_ok=True) 
with open(opt['save_path']+"opt.json", "w") as write_file:
    json.dump(opt, write_file, indent=4)


np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])


train_data,test_data,train_data_evaluation=LoadData(opt)
data_size=train_data.dataset[0][0].size()
if opt['rescale']:
    preprocess = lambda x: rescaling(dequantize(x))
    preprocess_inv = lambda x: quantize(rescaling_inv(x))
else:
    preprocess = lambda x: dequantize(x)
    preprocess_inv = lambda x: quantize(x)

if opt['learn']=='kl':
    model=Unet(in_channels=data_size[0], out_channels=data_size[0]*2, resolution=data_size[1]).to(opt['device'])
else:
    model=Unet(in_channels=data_size[0], out_channels=data_size[0], resolution=data_size[1]).to(opt['device'])
model.train()
optimizer=optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0.0,
                          betas=(0.9, 0.999), amsgrad=False,eps=0.00000001)


loss_list=[]
start_epoch=0
if opt['load_name']!=None:
    print('load:'+opt['load_name'])
    checkpoint = torch.load(opt['save_path']+opt['load_name']+'.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']+1
    loss_list=checkpoint['loss_list']
    print(len(loss_list))



for epoch in range(start_epoch, opt['epochs'] + 1):
    print('epoch:',epoch)
    for x, _ in tqdm(train_data):
        optimizer.zero_grad()
        if opt['data_set']=='MNIST':
            x=preprocess(x).view(-1,1,28,28).to(opt['device'])
        else:
            x=preprocess(x).to(opt['device'])
        noisy_x= (x+torch.randn_like(x)*opt['x_std']).requires_grad_()
        if opt['learn']=='kl':
            x_stats=model(noisy_x)
            loss=-Normal(x_stats[:,0:1,:,:],torch.exp(x_stats[:,1:2,:,:])).log_prob(x).sum([1,2,3]).mean(0)
        else:
            if opt['learn']=='energy':
                if opt['energy']=='sum':
                    energy=model(noisy_x).sum()
                elif opt['energy']=='ho':
                    energy=(((noisy_x-model(noisy_x))**2)).sum()
                else:
                    pass
                x_score=-grad(energy,noisy_x,create_graph=True)[0]

            elif opt['learn']=='score':
                x_score=model(noisy_x)
                
            denoised_x=noisy_x+opt['x_std']**2*x_score
            loss=torch.sum((denoised_x-x)**2,(1,2,3)).mean(0)

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    if  epoch%opt['save_freq']==0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list': loss_list,
            'opt': opt,
            }, 
            opt['save_path']+str(epoch)+'.pth'
            )

