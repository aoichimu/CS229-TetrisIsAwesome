#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:49:43 2016

@author: jiamingzeng
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os

class Agent:
    def __init__(self, model, memory, nb_step):
        