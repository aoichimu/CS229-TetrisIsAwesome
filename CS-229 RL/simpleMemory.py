#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Memory object intialized as memory = Memory(memorySize = n), where n is the 
desired size of the memory.

"""

from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np


class RingBuffer(object):
    def __init__(self, maxLength):
        self.maxLength = maxLength
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxLength)]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise KeyError()
        return self.data[index]

    def getLength(self):
        return self.length

    def append(self, experience):
        if self.length < self.maxLength:
            self.length += 1 
        elif self.length == self.maxLength:
            self.start = (self.start + 1) % self.maxLength

        self.data[(self.start + self.length - 1) % self.maxLength] = experience
        
class Memory(object):
    def __init__(self, memorySize):
        self.memorySize = memorySize
        
        #initiates memory 
        self.mem = RingBuffer(memorySize)
        
    def __getitem__(self, index):
        if index < 0 or index >= self.memorySize:
            raise KeyError()
        return self.mem[index]
        
    def append(self, state_t, action, reward, state_t1, term):
        self.mem.append(np.array([state_t, action, reward, state_t1, term]))
        
        
    def randSample(self, batch_size): #returns an array 
        x = sample_batch_ind(self.mem.getLength(), batch_size)
        y = np.zeros((batch_size, 5))
        for i in range(0,len(x)):
            y[i] = self.mem[x[i]]
        return y
        
        
def sample_batch_ind(memSize, minibatchSize):
    r = range(0, memSize)
    batch_indexes = random.sample(r, minibatchSize)
    return batch_indexes


    