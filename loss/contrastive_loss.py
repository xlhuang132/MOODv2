import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np   

def cosine_pairwise(x):
    x = x.permute((1, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-1)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise

def pairwise_similarity(outputs_1, outputs_2,temperature=0.5):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity
    '''  
    outputs=torch.cat((outputs_1,outputs_2),dim=0)
    B   = outputs.shape[0]
    outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    similarity_matrix = (1./temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))
    return similarity_matrix
 
 
def NT_xent(similarity_matrix):
    '''
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    ''' 

    N2  = len(similarity_matrix)
    N   = int(len(similarity_matrix) / 2)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2,N2)).cuda()

    NT_xent_loss        = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
    NT_xent_loss_total  = (1./float(N2)) * torch.sum(torch.diag(NT_xent_loss[0:N,N:]) + torch.diag(NT_xent_loss[N:,0:N]))

    return NT_xent_loss_total
 