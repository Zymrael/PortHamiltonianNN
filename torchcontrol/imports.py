# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:22:46 2019

@author: Zymieth
"""
import sys
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import torch.nn.functional as F
from operator import add
import numpy as np
import matplotlib.pyplot as plt