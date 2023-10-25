from torch.autograd import grad
import torch
import torch.nn as nn
from utils import *
from mmd import *
from networks import *
from torch import optim
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd.functional import jacobian
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def gibbs_sampler(x_init, backward_sampler, opt):
    x=x_init.to(opt['device'])
    samples=[]
    for i in tqdm(range(opt['gibbs_steps'])):
        noisy_x=x+torch.randn_like(x)*opt['noise_std']
        x=backward_sampler(noisy_x)
        samples.append(x.cpu())
    return torch.cat(samples, dim=0)

class DenoiserLearnedVar(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.x_dim=opt['x_dim']
        self.device=opt['device']
        self.net=FeedFowardNet(input_dim=self.x_dim, output_dim=self.x_dim*2,h_layer_num=opt['layer_num'],act=opt['act']).to(opt['device'])
        self.optimizer=optim.Adam(self.net.parameters(), lr=opt['lr'])

    def forward(self, noisy_x):
        mu,log_sigma=self.net(noisy_x).chunk(2,-1)
        sigma=torch.exp(log_sigma)
        return mu,sigma

    def logp_x_tx(self, x,noisy_x):
        noisy_x=noisy_x.to(self.device)
        x=x.to(self.device)
        mu,sigma=self.forward(noisy_x)
        return Normal(mu,sigma).log_prob(x).sum(1)
    
    def sample(self, noisy_x):
        with torch.no_grad():
            mu,sigma=self.forward(noisy_x)
        return Normal(mu,sigma).sample()
    





class DenoisingEBM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.x_dim=opt['x_dim']
        self.device=opt['device']
        self.noise_std=opt['noise_std']
        self.net=FeedFowardNet(input_dim=self.x_dim, output_dim=self.x_dim,h_layer_num=opt['layer_num'],act=opt['act']).to(opt['device'])
        self.optimizer=optim.Adam(self.net.parameters(), lr=opt['lr'])
        self.iso_cov=None



    def forward(self, noisy_x):
        noisy_x=noisy_x.requires_grad_()
        energy=self.net(noisy_x).sum()
        x_score=-grad(energy,noisy_x,create_graph=True)[0]
        denoised_x_mean=noisy_x+self.noise_std**2*x_score
        return denoised_x_mean
    
    def isotropic_cov_estimation(self, dataset):
        tx_dataset=(dataset+torch.randn_like(dataset)*self.noise_std).requires_grad_()
        energy_tx_dataset=self.net(tx_dataset.to(self.device)).sum()
        x_score_dataset=-grad(energy_tx_dataset,tx_dataset,retain_graph=True,create_graph=True)[0]
        with torch.no_grad():
            self.iso_cov=self.noise_std**2-self.noise_std**4*((x_score_dataset**2).sum(1).mean()/2)
        return self.iso_cov
        

    def logp_x_tx_isotropic_cov(self, x,noisy_x):
        noisy_x=noisy_x.to(self.device)
        x=x.to(self.device)
        if self.iso_cov is None:
            return "please estimate isotropic covariance first"
        x_mu=self.forward(noisy_x)
        log_prob=MultivariateNormal(x_mu,torch.diag(torch.ones_like(x_mu)*self.iso_cov)).log_prob(x)
        return log_prob.detach()
    
    def sample_isotropic_cov(self, noisy_x):
        if self.iso_cov is None:
            return "please estimate isotropic covariance first"
        x_mu=self.forward(noisy_x)
        return MultivariateNormal(x_mu,torch.diag(torch.ones_like(x_mu[0])*self.iso_cov)).sample()
       
    def get_hessian(self,noisy_x):
        def get_score_sum(noisy_x):
            energy=self.net(noisy_x).sum()
            score= grad(energy,noisy_x,retain_graph=True,create_graph=True)
            return -score[0].sum(0)
        return  jacobian(lambda a:  get_score_sum(a), noisy_x, vectorize=True).swapaxes(0, 1)


    def logp_x_tx_full_cov(self, x,noisy_x):
        x=x.to(self.device)
        noisy_x=noisy_x.view(1,2).to(self.device).requires_grad_()
        x_mu=self.forward(noisy_x)
        hessian_matrix = self.get_hessian(noisy_x)
        with torch.no_grad():
            x_cov=self.noise_std**4*hessian_matrix+torch.diag(torch.ones(2).to(self.device))*(self.noise_std)**2
            log_prob=MultivariateNormal(x_mu,x_cov).log_prob(x)
        return log_prob.detach()
    
    def sample_full_cov(self, noisy_x):
        noisy_x=noisy_x.view(1,2).requires_grad_()
        x_mu=self.forward(noisy_x)
        hessian_matrix = self.get_hessian(noisy_x)
        with torch.no_grad():
            x_cov=self.noise_std**4*hessian_matrix+torch.diag(torch.ones(2).to(self.device))*(self.noise_std)**2
            return MultivariateNormal(x_mu,x_cov).sample()

