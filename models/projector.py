import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, model_name=None,expansion=4,alg_name=''):
        super(Projector, self).__init__()
        if model_name: 
            if model_name == 'WRN_28_2':
                self.linear_1 = nn.Linear(128, 128)
                self.linear_2 = nn.Linear(128, 128)
            elif model_name == 'WRN_28_8':
                self.linear_1 = nn.Linear(512, 128)
                self.linear_2 = nn.Linear(128, 128)
            elif model_name == 'Resnet34':
                self.linear_1 = nn.Linear(512, 128)
                self.linear_2 = nn.Linear(128, 128)
            elif model_name == 'Resnet50':
                self.linear_1 = nn.Linear(2048, 512)
                self.linear_2 = nn.Linear(512, 128) 
        elif expansion == 0:
            self.linear_1 = nn.Linear(128, 128)
            self.linear_2 = nn.Linear(128, 128)
        else:
            self.linear_1 = nn.Linear(512*expansion, 512)
            self.linear_2 = nn.Linear(512, 128)

    def forward(self, x, internal_output_list=False,normalized=False):          
        output = self.linear_1(x)
        output = F.relu(output) 
        if normalized:
            output = F.normalize(self.linear_2(output),dim=-1)
        else:
            output = self.linear_2(output) 

        return output 