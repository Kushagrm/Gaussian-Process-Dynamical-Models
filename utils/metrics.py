#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing metrics for GPLVM models

- Test Reconstruction error for Y_train with and without missing data 
- Negative test log-likelihood

"""

import torch
import numpy as np
#from pyro.distributions.transforms import AffineTransform

def float_tensor(X): return torch.tensor(X).float()

def rmse(Y_test, Y_recon):
    
    return torch.mean((Y_test - float_tensor(Y_recon))**2).sqrt()

def rmse_missing(Y_test, Y_recon):
    
    return torch.sqrt(torch.Tensor([np.nanmean(np.square(Y_test - Y_recon))]))

def test_nlpd(model, likelihood, X_test, Y_test, model_name):
    
    if model_name in ['point', 'map']:
                    
            preds_f_star = model(X_test.X.cuda())
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
            
            gaussian = torch.distributions.MultivariateNormal(X_test.loc, X_test.covariance_matrix)
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
    return nlpd_test
    

# def decomm_test_log_likelihood(model, Y_test, test_dist):
    
#      if isinstance(model.enc_flow.flows[0],  AffineTransform):
#          model.decoder.X = model.enc_base.mu.detach()
#      else:
#          model.decoder.X = model.enc_flow.X_map(n_restarts=1,use_base_mu=True)
     
#      test_log_lik_samples = torch.Tensor([model.log_p_of_y_given_x(test_dist(), Y_test) for _ in range(100)])
#      return torch.mean(test_log_lik_samples)/len(Y_test)

