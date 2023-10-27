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
from torchvision.utils import save_image
from torch.autograd import grad



def diag_hessian_approximation_score(score_model,batch_x,sample_num):
    bs=batch_x.size(0)
    batch_x=batch_x.requires_grad_()
    diag_hessian=torch.zeros_like(batch_x).view(bs,-1)
    for i in range(0,sample_num):
        v=torch.bernoulli(torch.ones_like(diag_hessian)*0.5)*2-1
        score=score_model(batch_x)
        Sv=torch.bmm(score.view(bs, 1, -1), v.view(bs, -1, 1)).sum()
        Hv=grad(Sv,batch_x,retain_graph=False,create_graph=False)[0].view(bs,-1)
        with torch.no_grad():
            diag_hessian+=v*Hv
    return diag_hessian/sample_num


def diag_hessian_approximation_energy(energy_model,batch_x,sample_num,opt):
    bs=batch_x.size(0)
    batch_x=batch_x.requires_grad_()
    diag_hessian=torch.zeros_like(batch_x).view(bs,-1)
    for i in range(0,sample_num):
        v=torch.bernoulli(torch.ones_like(diag_hessian)*0.5)*2-1
        if opt['energy']=='sum':
            energy=energy_model(batch_x).sum()
        elif opt['energy']=='ho':
            energy=((batch_x-energy_model(batch_x))**2).sum()
        else:
            raise NotImplementedError
        
        score=-grad(energy,batch_x,retain_graph=True,create_graph=True)[0].view(bs,-1).requires_grad_()
        Sv=torch.bmm(score.view(bs, 1, -1), v.view(bs, -1, 1)).sum()
        Hv=grad(Sv,batch_x,retain_graph=False,create_graph=False)[0].view(bs,-1)
        with torch.no_grad():
            diag_hessian+=v*Hv
    return diag_hessian/sample_num


def sample_batch_img(imgs,img_ids,path):
    for img,img_id in zip(imgs,img_ids):
        save_image(img, os.path.join(path, 'image_{}.png'.format(img_id)))
    return  None  
                
def generating_samples(model,num,opt,s_opt):
    if opt['rescale']:
        preprocess = lambda x: rescaling(dequantize(x))
        preprocess_inv = lambda x: quantize(rescaling_inv(x))
    else:
        preprocess = lambda x: dequantize(x)
        preprocess_inv = lambda x: quantize(x)

    def generating_batch(model,opt,s_opt,batch_size,batch_id):
        img_ids=np.arange(0,batch_size)+batch_id*s_opt['sample_bs']
        with torch.no_grad():
            x=s_opt['init_mu']+s_opt['init_std']*torch.randn(batch_size,opt['in_channel'],opt['res'],opt['res']).to(opt['device'])
            x_mu=preprocess(x)
        img_list=[]
        for e in tqdm(range(0,s_opt['gibbs_steps']+1)):
            if e%s_opt['save_freq']==0:
                if s_opt['task']=='save_mean':
                    e_save_path=s_opt['save_path']+'/mean/'+str(e)+'/'
                    os.makedirs(e_save_path, exist_ok=True) 
                    sample_batch_img(preprocess_inv(x_mu).cpu(),img_ids,e_save_path)
                    img_list.append(preprocess_inv(x_mu).cpu())
                elif s_opt['task']=='save_sample':
                    e_save_path=s_opt['save_path']+'/sample/'+str(e)+'/'
                    os.makedirs(e_save_path, exist_ok=True) 
                    sample_batch_img(preprocess_inv(x).cpu(),img_ids,e_save_path)
                    img_list.append(preprocess_inv(x).cpu())
                elif s_opt['task']=='return_mean':
                    img_list.append(preprocess_inv(x_mu).cpu())
                elif s_opt['task']=='return_sample':
                    img_list.append(preprocess_inv(x).cpu())
                else:
                    raise NotImplementedError
            if opt['learn']=='energy':
                if opt['energy']=='sum':
                    energy=model(noisy_x).sum()
                elif opt['energy']=='ho':
                    energy=(((noisy_x-model(noisy_x))**2)).sum()
                else:
                    raise NotImplementedError
                noisy_x=(x+torch.randn_like(x)*opt['x_std']).requires_grad_()
                score= -grad(energy,noisy_x,retain_graph=False,create_graph=False)[0]
                with torch.no_grad():
                    x_mu=noisy_x+opt['x_std']**2*score
                x_cov_diag=opt['x_std']**4*diag_hessian_approximation_energy(model,noisy_x,s_opt['trace_hessian_it'],opt)+opt['x_std']**2
                x=x_mu+torch.randn_like(x_mu)*torch.sqrt(torch.relu(x_cov_diag)).view_as(x_mu)
            
            elif opt['learn']=='score':
                with torch.no_grad():
                    noisy_x=(x+torch.randn_like(x)*opt['x_std'])
                    score=model(noisy_x)
                    x_mu=noisy_x+opt['x_std']**2*score
                x_cov_diag=opt['x_std']**4*diag_hessian_approximation_score(model,noisy_x,s_opt['trace_hessian_it'])+opt['x_std']**2
                x=x_mu+torch.randn_like(x_mu)*torch.sqrt(torch.relu(x_cov_diag)).view_as(x_mu)

        return img_list


    model.eval()

    if s_opt['task'] in ['return_sample', 'return_mean']:
        assert num==s_opt['sample_bs']

    total_batches=int(num/s_opt['sample_bs'])
    for batch_id in range(0,total_batches):
        print('pregress:',batch_id,'/',total_batches)
        img_list=generating_batch(model,opt,s_opt,s_opt['sample_bs'],batch_id)
    if num%s_opt['sample_bs']>0:
        batch_id+=1
        img_list=generating_batch(model,opt,s_opt,num%s_opt['sample_bs'],batch_id)

    return img_list








