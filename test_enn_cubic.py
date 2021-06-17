# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:32:03 2021

@author: minhh
"""

import numpy as np
from nn.enn import EvidentialNN
import os


def generate_data(x, noise=False, sigma_coeff=3):
    x = x.astype(np.float32)
    # y = np.sin(3*x) / (3*x)
    y = x ** 3
    
    if noise:
        sigma = sigma_coeff * np.ones_like(x)
    else:
        sigma = np.zeros_like(x)
        
    r = np.random.normal(0, sigma).astype(np.float32)
    return y+r, sigma

    
    
if __name__ == "__main__":
    savepath = os.getcwd()
    net = EvidentialNN(input_dim=1, output_dim=1, lam=1e-2)
    
    net.buffer_train = net.fill_buffer(xlimits=[[-3, 3]], evaluate_func=generate_data)
    net.buffer_test = net.fill_buffer(xlimits=[[-3, 3]], evaluate_func=generate_data, n=300)
    
    scale = True
    model, rmse, nll = net.train(iters=300, bsize=125, verbose=True, initial_training=False, scale=scale)