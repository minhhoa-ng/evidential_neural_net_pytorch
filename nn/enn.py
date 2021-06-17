# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:08:52 2021

@author: minhh
"""

import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from nn.buffer import ReplayBuffer

import datetime, time
import matplotlib.pyplot as plt
from scipy.stats import norm

##
class Softplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 6) -> None:
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        core = self.beta * input
        core[core > self.threshold] = self.threshold
        res = 1/ self.beta * torch.log(1 + torch.exp(core))
        return res

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)    



#--------------------
# output layers
class LayerNormal(nn.Module):
    def __init__(self, layer):
        super(self, LayerNormal).__init__()
        self.layer = layer
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        output = self.layer(x)
        mu, logsigma = torch.tensor_split(output, 2, dim=-1)
        sigma = self.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)
        

class LayerNormalGamma(nn.Module):
    def __init__(self, layer):
        super(LayerNormalGamma, self).__init__()
        self.layer = layer
        self.softplus = nn.Softplus()
        
    def evidence(self, x):
        return self.softplus(x)
    
    def forward(self, x):
        output = self.layer(x)     
        mu, logv, logalpha, logbeta = torch.tensor_split(output, 4, dim=-1)
        # print('forward', torch.isnan(logv).sum(), torch.isnan(logalpha).sum(), torch.isnan(logbeta).sum())
        # print('forward', logv.sum(), logalpha.sum(), logbeta.sum())
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)
    
    
class LayerDirichlet(nn.Module):
    def __init__(self, layer):
        super(LayerDirichlet, self).__init__()
        self.layer = layer
        
    def forward(self, x):
        output = self.layer(x)
        evidence = torch.exp(output)
        alpha = evidence + 1
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return torch.cat([alpha, prob], dim=-1)
    
    
class LayerSigmoid(nn.Module):
    def __init__(self, layer):
        super(LayerSigmoid, self).__init__()
        self.layer = layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        logits = self.layer(x)
        prob = self.sigmoid(logits)
        return [logits, prob]
        

#---------------------------
# overall enn architecture
last_layers = {'normal': [LayerNormal, 2],
               'normalgamma': [LayerNormalGamma, 4],
               'drichlet': [LayerDirichlet, 1],
               'sigmoid': [LayerSigmoid, 1]}


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim, last_layer='', hidden_dims=[300, 200]):
        super(MLP, self).__init__()
        assert last_layer != '', "please assign the last layer's architecture"
        model, n_params = last_layers[last_layer]
        layers = OrderedDict([('input_linear', nn.Linear(input_dim, hidden_dims[0])),
                              ('input_activation', nn.ReLU()),
                              ('hidden_linear', nn.Linear(hidden_dims[0], hidden_dims[1])),
                              ('hidden_activation', nn.ReLU()),
                              ('evidental_layer', model(nn.Linear(hidden_dims[1], output_dim*n_params)))])
        self.module = nn.Sequential(layers)
         
    def forward(self, inputs):
        return self.module(inputs)
        
#---------------
# loss functions
def MSE(y, y_, reduce=True):
    ax = list(range(1, len(y.shape)))
    mse = torch.mean((y-y_)**2, dim=ax)
    return torch.mean(mse) if reduce else mse


def RMSE(y, y_):
    rmse = torch.sqrt(torch.mean((y-y_)**2))
    return rmse


def Gaussian_NLL(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))
    logprob = - torch.log(sigma) - 0.5 * torch.log( 2 * np.pi) - 0.5 * ((y-mu)/sigma)**2
    loss = torch.mean(-logprob, dim=ax)
    return torch.mean(loss) if reduce else loss
    
    
def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))
    
    log_likelihood = 0.5 * (-torch.exp(-logvar) * (mu-y) ** 2 - 
                            torch.log(2 * torch.tensor(np.pi)) - 
                            logvar)
    loss = torch.mean(-log_likelihood, dim=ax)
    return torch.mean(loss) if reduce else loss


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1+v)
    pi = torch.tensor(np.pi)
    nll = 0.5 * torch.log(pi/v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y-gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    return torch.mean(nll) if reduce else nll


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1-1)/b1 * (v2*torch.square(mu2-mu1)) \
        + 0.5 * v2/v1 \
        - 0.5 * torch.log( torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2*torch.log(b1/b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1/b1
    return KL


def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = torch.abs(y-gamma)
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error * kl
    else:
        evi = 2 * alpha + v
        reg = error * evi
    return torch.mean(reg) if reduce else reg


def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta  = torch.tensor_split(evidential_output, 4, dim=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg
    
    
# ---------------
# Evidential NN
class EvidentialNN:
    def __init__(self, 
                 input_dim, # no of decision vars
                 output_dim, # no of objectives
                 hidden_dims=[200, 300],
                 learning_rate=5e-4, 
                 lam=1., # value lam is very important, the more noise, the smaller lam
                 epsilon=1e-2, 
                 maxi_rate=1e-5, 
                 tag=""):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = MLP(input_dim, output_dim, last_layer='normalgamma', hidden_dims=hidden_dims)
        self.nll_loss = NIG_NLL
        self.reg_loss = NIG_Reg
        self.lr = learning_rate
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        self.lam = lam
        self.mx_rate = maxi_rate
        self.eps = epsilon
        self.min_rmse = self.running_rmse = float('inf')
        self.min_nll = self.running_nll = float('inf')
        self.min_vloss = self.running_vloss = float('inf')
        self.device = 'cpu'
        print(self.model)
    
    
    def loss_function(self, y, mu, v, alpha, beta, reduce=True, return_comps=False):
        nll_loss = self.nll_loss(y, mu, v, alpha, beta, reduce=reduce)
        reg_loss = self.reg_loss(y, mu, v, alpha, beta, reduce=reduce)
        loss = nll_loss + self.lam * reg_loss 
        return (loss, (nll_loss, reg_loss)) if return_comps else loss
    
    
    def run_train_step(self, x, y, initial_training=False):
        y_ = self.model(x)
        mu, v, alpha, beta = torch.tensor_split(y_, 4, dim=-1)
        loss, (nll_loss, reg_loss) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)
        
        torch.autograd.set_detect_anomaly(True)
        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 2)
        self.optim.step()        
        
        # print('weight linear', self.model.module[4].layer.weight.grad)
        
        # if not initial_training:
        #     self.lam -= self.mx_rate * (reg_loss.detach().numpy().mean() - self.eps)
        return loss.item(), nll_loss.item(), reg_loss.item(), \
                mu.detach().numpy(), v.detach().numpy(), \
                alpha.detach().numpy(), beta.detach().numpy()
    
    
    def evaluate_enn(self, x, y):
        outputs = self.model(x)
        mu, v, alpha, beta = torch.tensor_split(outputs, 4, dim=-1)

        rmse = RMSE(y, mu)
        loss, (nll, reg_loss) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

        return mu.detach().numpy(), v.detach().numpy(), \
                alpha.detach().numpy(), beta.detach().numpy(), \
                loss.item(), rmse.item(), nll.item(), reg_loss.item()
    
    
    def fill_buffer(self, xlimits=[[-4, 4]], evaluate_func=None, n=1000):
        assert evaluate_func is not None, 'Evaluation function not assigned'
        xs = np.concatenate([np.linspace(xmin, xmax, n) for (xmin, xmax) in xlimits]).reshape(-1,1)
        ys, sigma = evaluate_func(xs, noise=True)
        buffer = ReplayBuffer(self.input_dim,
                              self.output_dim,
                              var_limits=np.array(xlimits))
        buffer.store(xs, ys)        
        return buffer
    
    
    def EI(self, mu, epistemic, f_min):
        if epistemic.size == 1:
            return np.array([0.0])    
        ei = np.zeros_like(epistemic)
        cond = epistemic > 0
        args0 = (f_min - mu[cond])/np.sqrt(epistemic[cond])
        args1 = (f_min - mu[cond])*norm.cdf(args0)
        args2 = np.sqrt(epistemic[cond])*norm.pdf(args0)
        ei[cond] = args1 + args2
        return ei


    def is_row_in_array(self, rows, arr):
        if rows.shape[0] > 1:
            return np.array([(arr == r).all(axis=1).any() for r in rows])
        else:
            return (arr == rows).all(axis=1).any()
    
    
    def update_running(self, previous, current, alpha=0.0):
        if previous == float('inf'):
            new = current
        else:
            new = alpha*previous + (1-alpha)*current
        return new    
    
    
    def train(self, iters=10000, bsize=125, verbose=True, initial_training=True, scale=False, scale_factor=100):
        tic = time.time()
        check = initial_training * 50 + (1- initial_training) * 50
        
        for i in range(iters):
            xs, ys = (x.to(self.device) for x in 
                      self.buffer_train.sample(batch_size=bsize, scale=scale))
            loss, nll_loss, reg_loss, y_hat, v, alpha, beta = self.run_train_step(xs, ys, initial_training=initial_training)

            if i % check == 0:
                xs_test, ys_test = (x.to(self.device) for x in 
                                    self.buffer_test.sample(batch_size=bsize, scale=scale))
                mu, v, alpha, beta, vloss, rmse, nll, reg_loss = self.evaluate_enn(xs_test, ys_test)
                xs_test = xs_test.detach().numpy()
                ys_test = ys_test.detach().numpy()
                
                # push data points with high epistemic uncertainty into the training buffer
                if not initial_training:
                    if scale:
                        mu = self.buffer_train._unscale(x=mu, name='obj')
                        xs_test = self.buffer_train._unscale(x=xs_test, name='var')
                        ys_test = self.buffer_train._unscale(x=ys_test, name='obj')
                    epistemic = np.sqrt(beta / (v * (alpha - 1)))[:,0]*scale_factor
                    # print('epistemic', epistemic.mean(), epistemic.std())
                    ei = self.EI(mu[:,0], epistemic, f_min=ys_test.min().item())
                    # added = epistemic.argsort()[::-1][:20]
                    # unique = 1 - self.is_row_in_array(rows=xs_test.detach().numpy(), arr=self.buffer_train.var_buf[:self.buffer_train.size])
                    added = ei.argsort()[::-1][:20].flatten()
                    self.buffer_train.store(xs_test[added], ys_test[added])

                self.running_rmse = self.update_running(self.running_rmse, rmse)
                if self.running_rmse < self.min_rmse:
                    self.min_rmse = self.running_rmse

                self.running_nll = self.update_running(self.running_nll, nll)
                if self.running_nll < self.min_nll:
                    self.min_nll = self.running_nll

                self.running_vloss = self.update_running(self.running_vloss, vloss)
                if self.running_vloss < self.min_vloss:
                    self.min_vloss = self.running_vloss
                
                self.plot_evi(i=str(i)+ str('improve') * (1 - initial_training), scale=scale)
                
            if verbose and i % 10 == 0: 
                print("[{}]  RMSE: {:.4f} \t NLL: {:.4f} \t loss: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f} \t t: {:.2f} sec"
                      .format(i, float(self.min_rmse), nll, vloss, reg_loss, self.lam, time.time()-tic))
                tic = time.time()
        
        return self.model, self.min_rmse, self.min_nll
    
    
    def plot_predictions(self, x_train, y_train, x_test, y_test, y_pred, n_stds=5, kk=0, iteration='',
                         train_limits=[-4, 4], test_limits=[-5, 5], xlim=[-5, 5], ylim=[-50, 50], scale_factor=100):
        x_test = x_test[:, 0]
        mu, v, alpha, beta = np.split(y_pred, 4, axis=-1)
        mu = mu[:, 0]
        var = np.sqrt(beta / (v * (alpha - 1)))*scale_factor # epistemic
        var = np.minimum(var, 1e3)[:, 0]  # for visualization
    
        plt.figure(figsize=(5, 3), dpi=200)
        plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
        plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
        plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
        plt.plot([train_limits[0]]*2, ylim, 'k--', alpha=0.4, zorder=0)
        plt.plot([train_limits[1]]*2, ylim, 'k--', alpha=0.4, zorder=0)
        for k in np.linspace(0, n_stds, 4):
            plt.fill_between(
                x_test, (mu - k * var), (mu + k * var),
                alpha=0.3,
                edgecolor=None,
                facecolor='#00aeef',
                linewidth=0,
                zorder=1,
                label="Unc." if k == 0 else None)
        plt.gca().set_ylim(ylim)
        plt.gca().set_xlim(xlim)
        plt.legend(loc="upper left")
        plt.title('Iteration ' + iteration)
        plt.show()
        return var
        
        
    def plot_evi(self, i='', save="evi", ext=".pdf", scale=False):
        x_train=self.buffer_train.var_buf[:self.buffer_train.size]
        y_train=self.buffer_train.obj_buf[:self.buffer_train.size]
        x_test=self.buffer_test.var_buf[:self.buffer_test.size]
        y_test=self.buffer_test.obj_buf[:self.buffer_test.size]
        
        scaled_x_test = self.buffer_train._scale(x=x_test, name='var')
        y_pred = self.model(torch.as_tensor(scaled_x_test, dtype=torch.float32).to(self.device)).detach().numpy()
        # print('pred_scaled', y_pred.mean(axis=0).astype(float), y_pred.std(axis=0).astype(float))
        if scale:
            y_pred[:,0] = self.buffer_train._unscale(x=y_pred[:,0], name='obj') 
        # print('pred_unscaled', y_pred.mean(axis=0).astype(float), y_pred.std(axis=0).astype(float))
        # print('test', y_test.mean(axis=0).astype(float), y_test.std(axis=0).astype(float))
        var = self.plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0, iteration=i)   
        return x_train, y_train, x_test, y_test, y_pred, var    
    
    
    
