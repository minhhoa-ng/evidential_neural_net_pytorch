# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:04:01 2021

@author: minhh
"""
import numpy as np
import torch




# Buffer storing samples for training ANN
class ReplayBuffer:
    def __init__(self, ndim, nobj, var_limits, maxlen=int(10e3)):
        self.size, self.ptr = 0, 0
        self.maxlen = maxlen
        self.var_buf = np.zeros((maxlen, ndim),
                                dtype=np.float32)
        self.obj_buf = np.zeros((maxlen, nobj),
                                dtype=np.float32)
        self.min_obj = np.zeros((1, nobj),
                                dtype=np.float32)
        self.max_obj = np.zeros((1, nobj),
                                dtype=np.float32)
        self.min_var = var_limits[:, 0:1].T
        self.max_var = var_limits[:, 1:2].T

    def __len__(self):
        return self.size

    def is_full(self):
        return self.__len__() == self.maxlen

    def store(self, var, obj):
        if type(var) == np.ndarray:
            n = min(len(var), self.maxlen-self.ptr)
        else:
            n = 1
        self.var_buf[self.ptr: self.ptr+n] = var[:n]
        self.obj_buf[self.ptr: self.ptr+n] = obj[:n]

        self.ptr = (self.ptr + n) % self.maxlen
        self.size = min(self.size + n, self.maxlen)
        self._get_min_max()
        
    def _get_min_max(self):
        self.min_obj[0] = self.obj_buf.min(axis=0)
        self.max_obj[0] = self.obj_buf.max(axis=0)
        
    def _scale(self, x=None, name=''):
        if name == 'var':
            if x is None:
                x = self.var_buf
            mn = self.min_var
            mx = self.max_var
        else:
            if x is None:
                x = self.obj_buf
            mn = self.min_obj
            mx = self.max_obj
        x = (x - mn)/(mx - mn + 1e-5)
        return x
    
    def _unscale(self, x=None, name=''):
        if name == 'var':
            if x is None:
                x = self.var_buf
            mn = self.min_var
            mx = self.max_var
        else:
            if x is None:
                x = self.obj_buf
            mn = self.min_obj
            mx = self.max_obj
        x = x*(mx - mn + 1e-5) + mn
        return x        
    
    def _encode_sample(self, idxes, scale=True):
        if scale:
            return (torch.as_tensor(x[idxes], dtype=torch.float32)
                    for x in (self._scale(name='var'), 
                              self._scale(name='obj')))
        else:
            return (torch.as_tensor(x[idxes], dtype=torch.float32)
                    for x in (self.var_buf, self.obj_buf))

    def sample(self, batch_size=32, scale=True):
        idxes = np.random.randint(0, self.size, size=batch_size)
        return self._encode_sample(idxes, scale=scale)
    
    
    def copy(self):
        new = ReplayBuffer(self.var_buf.shape[1], 
                           self.obj_buf.shape[1], 
                           np.concatenate((self.min_var, self.max_var), axis=1))
        new.__dict__ = self.__dict__
        return new
      
