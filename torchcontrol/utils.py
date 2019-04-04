# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 09:25:22 2019

@author: Zymieth
"""
import torch
device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device('cuda')

def genpoints(xmin,xmax,ymin,ymax,number_points):
    xx = torch.linspace(xmin,xmax,number_points)
    yy = torch.linspace(ymin,ymax,number_points)
    c = 1
    P = []
    for i in range(number_points):
        for j in range(number_points):
            P.append([xx[i],yy[j]])
    return torch.Tensor(P).to(device)

def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)