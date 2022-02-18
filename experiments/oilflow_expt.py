#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for experiments with oilflow data

4 different inference modes:
    
   models = ['point','map','gaussian','nn_gaussian']

"""

from utils.data import load_real_data 
import pickle as pkl
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable, VariationalLatentVariable, NNEncoder
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class OilFlowModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(OilFlowModel, self).__init__(X, q_f)
        
        # Kernel 
        #self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

     def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
     
if __name__ == '__main__':
    
    TEST = True
    increment = np.random.randint(0,100,5)

model_dict = {}
noise_trace_dict = {}

for k in range(4):
    
    SEED = 7 + increment[k]
    torch.manual_seed(SEED)

    # Load some data
    
    N, d, q, X, Y, labels = load_real_data('oilflow')
    
    Y_train, Y_test = train_test_split(Y.numpy(), test_size=0.2, random_state=SEED)
    lb_train, lb_test = train_test_split(labels, test_size=0.2, random_state=SEED)
    
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    
    # Setting shapes
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = 10
    n_inducing = 25
    pca = False
    
    # Run all 4 models and store results
    
    #models = ['point','map','gauss', 'nn_gauss']
    models = ['gauss']
    steps = [20000, 20000, 20000, 20000]
    steps_per_model = dict(zip(models, steps))
    
    for model_name in models:
        
        # Define prior for X
        X_prior_mean = torch.zeros(N, latent_dim)  # shape: N x Q
        X_prior_mean_test = X_prior_mean[0:len(Y_test),:]
    
        # Initialise X with PCA or 0s.
        if pca == True:
              X_init = _init_pca(Y_train, latent_dim) # Initialise X to PCA 
        else:
              X_init = torch.nn.Parameter(torch.zeros(N, latent_dim))
        
        # Each inference model differs in its latent variable configuration / 
        # LatentVariable (X)
        
        # defaults - if a model needs them they are internally assigned
        nn_layers = None
        prior_x = None
        prior_x_test = None
        
        if model_name == 'point':
            
            ae = False
            X = PointLatentVariable(X_init)
            
        elif model_name == 'map':
                        
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test))
            X = MAPLatentVariable(X_init, prior_x)
            
        elif model_name == 'gauss':
            
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test))
            X = VariationalLatentVariable(X_init, prior_x, latent_dim)
        
        elif model_name == 'nn_gauss':
            
            ae = True
            nn_layers = (10,5)
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1]))
            prior_x_test = MultivariateNormalPrior(X_prior_mean_test, torch.eye(X_prior_mean.shape[1]))
            X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
            
            
        # Initialise model, likelihood, elbo and optimizer
        
        model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers)
        likelihood = GaussianLikelihood()
        elbo = VariationalELBO(likelihood, model, num_data=len(Y_train))
    
        optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
        ], lr=0.001)
    
        # Model params
        print(f'Training model params for model {model_name}')
        model.get_trainable_param_names()
        
        #### Transfer on to GPU if available
        
        # if torch.cuda.is_available():
        #     device = 'cuda'
        #     model = model.cuda()
        #     likelihood = likelihood.cuda()
        #     Y_train = Y_train.cuda()
        #     Y_test = Y_test.cuda()
        # else:
        #     device = 'cpu'

        # print('The device is ' + device)
            
        # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
        # using the optimizer provided.
        
        loss_list = []
        noise_trace = []
        
        #iterator = trange(steps_per_model[model_name], leave=True)
        iterator = trange(600)
        batch_size = 100
        for i in iterator: 
            batch_index = model._get_batch_idx(batch_size)
            optimizer.zero_grad()
            if model_name in ['point','map', 'gauss']:
                sample = model.sample_latent_variable()  # a full sample returns latent x across all N
            else:
                sample = model.sample_latent_variable(Y_train)
            sample_batch = sample[batch_index]
            output_batch = model(sample_batch)
            loss = -elbo(output_batch, Y_train[batch_index].T).sum()
            loss_list.append(loss.item())
            noise_trace.append(np.round(likelihood.noise_covar.noise.item(),3))
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
        model.store(loss_list, likelihood)
            
        # Save models & training info
        
        #print(model.covar_module.base_kernel.lengthscale)
        #model_dict[model_name + '_' + str(SEED)] = model
        #noise_trace_dict[model_name + '_' + str(SEED)] = noise_trace
        
        ### Saving training report
        
        # from utils.visualisation import *
        
        X_train_mean = model.get_X_mean(Y_train)
        X_train_scales = model.get_X_scales(Y_train)
        
        #plot_report(model, loss_list, lb_train, colors=['r', 'b', 'g'], save=f'oilflow_{model_name}_{SEED}', X_mean=X_train_mean, X_scales=X_train_scales, model_name=model.X.__class__.__name__)
        
        # #### Saving model with seed 
        # print(f'Saving {model_name} {SEED}')
        
        # filename = f'oilflow_{model_name}_{SEED}_trained.pkl'
        # with open(f'pre_trained_models/oilflow/{filename}', 'wb') as file:
        #     pkl.dump(model.state_dict(), file)

        ####################### Testing Framework ################################################
        if TEST:
        #Compute latent test & reconstructions
            with torch.no_grad():
                model.eval()
                likelihood.eval()
            
            if ae is True:
                X_test_mean, X_test_covar = model.predict_latent(Y_train, 
                                                                  Y_test, 
                                                                  optimizer.defaults['lr'], 
                                                                  likelihood, 
                                                                  SEED,
                                                                  prior_x=prior_x_test, 
                                                                  ae=True, 
                                                                  model_name='nn_gauss', 
                                                                  pca=pca)
             
            
            else: # either point, map or gauss
                losses_test,  X_test = model.predict_latent(Y_train, Y_test, optimizer.defaults['lr'], 
                                              likelihood, SEED, prior_x=prior_x_test, ae=ae, 
                                              model_name=model_name,pca=pca)
                    
            
            # Compute training and test reconstructions
            if model_name in ('point', 'map'):
                    X_test_mean = X_test.X
            elif model_name == 'gauss':
                   X_test_mean = X_test.q_mu.detach().cpu()
       
            Y_test_recon, Y_test_pred_covar = model.reconstruct_y(torch.Tensor(X_test_mean), Y_test, ae=ae, model_name=model_name)
            Y_train_recon, Y_train_pred_covar = model.reconstruct_y(torch.Tensor(X_train_mean), Y_train, ae=ae, model_name=model_name)
            
            # ################################
            # Compute the metrics:
            ##################################
            
            from utils.metrics import *
            
            # 1) Reconstruction error - Train & Test
            
            rmse_train = rmse(Y_train, Y_train_recon.T)
            rmse_test = rmse(Y_test, Y_test_recon.T)
            
            print(f'Train Reconstruction error {model_name} = ' + str(rmse_train))
            print(f'Test Reconstruction error {model_name} = ' + str(rmse_test))
            
            # 2) Test NLPD
            
            if model_name in ['point', 'map']:
                
                preds_f_star = model(X_test.X)
                preds_y_star = likelihood(preds_f_star)
                nlpd_test = -preds_y_star.log_prob(Y_test.T).sum()/len(Y_test)
                
            elif model_name in ['gauss']:
                
                nlpds = []
                for i in range(100):
                    
                    X_test_sample_per_point =  X_test.forward()
                    preds_f_star = model(X_test_sample_per_point)
                    preds_y_star = likelihood(preds_f_star)
                    nlpd_sample = -preds_y_star.log_prob(Y_test.T).sum()/len(Y_test)
                    nlpds.append(nlpd_sample)
                
                nlpds = [x.cpu().detach().item() for x in nlpds]
                nlpd_test = np.mean(nlpds)
                
            elif model_name in ['nn_gauss']:
                
                gaussian = torch.distributions.MultivariateNormal(X_test_mean, X_test_covar)
                nlpds = []
                for i in range(100):
                    
                    X_test_sample_per_point =  gaussian.rsample()
                    preds_f_star = model(X_test_sample_per_point)
                    preds_y_star = likelihood(preds_f_star)
                    nlpd_sample = -preds_y_star.log_prob(Y_test.T).sum()/len(Y_test)
                    nlpds.append(nlpd_sample)
                
                nlpds = [x.cpu().detach().item() for x in nlpds]
                nlpd_test = np.mean(nlpds)
                
            print(f'Test NLPD {model_name} = ' + str(nlpd_test))
    
    
    
    
    
