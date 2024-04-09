import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from models.projector import  Projector 
from .classifier import Classifier
import torch.nn.functional as F
from torchvision.models.resnet import resnet34,resnet50
__all__ = ['ResNet_s', 'Resnet20', 'Resnet32','Resnet34','Resnet50','Resnet44', 'Resnet56', 'Resnet110', 'Resnet1202',]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.fc = NormedLinear(64, num_classes) 
        else:
            self.fc = nn.Linear(64, num_classes) 
        self.apply(_weights_init) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x,**kwargs):    
        if 'classifier' in kwargs:
            return self.fc(x)   
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        encoding = out.view(out.size(0), -1) 
        out = self.fc(encoding) 
        if 'return_encoding'in kwargs:
            return out, encoding
        else:
            return out

class Resnet34_(nn.Module):
    def __init__(self,num_classes=10, feature_dim=64,fc2_enable=False,fc2_out_dim=None):
        super(Resnet34_, self).__init__()

        self.f = []
        for name, module in resnet34().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module) 
        self.encoder = nn.Sequential(*self.f) 
        self.projector=Projector(model_name='Resnet34')
        self.fc=nn.Linear(512, num_classes, bias=False)
        self.fc2_enable=fc2_enable
        out_features =512
        assert not self.fc2_enable or self.fc2_enable and fc2_out_dim!=None
        if self.fc2_enable:
            self.fc2 = Classifier(out_features, fc2_out_dim) 
    
    def forward(self, x,return_encoding=False,return_projected_feature=False,classifier=False,training=True):
        if return_projected_feature: 
            return self.projector(x)
        if classifier:
            if self.fc2_enable:
                return self.fc(x),self.fc2(x)
            return self.fc(x)  
        feature=self.encoder(x)
        encoding = feature.view(feature.size(0), -1) 
        if return_encoding:            
            return encoding
        out=self.fc(encoding) 
        if not training:
            return out
        if self.fc2_enable: 
            out2=self.fc2(encoding)
            return out,out2
        return out
 
class Resnet50_(nn.Module):
    def __init__(self,num_classes=100, feature_dim=128,fc2_enable=False,fc2_out_dim=None):
        super(Resnet50_, self).__init__()
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module) 
        self.f = nn.Sequential(*self.f)
        
        self.projector=Projector(model_name='Resnet50')
        self.fc2_enable=fc2_enable 
        self.fc=nn.Linear(2048, num_classes, bias=True) 
        out_features =2048
        assert not self.fc2_enable or self.fc2_enable and fc2_out_dim!=None
        if self.fc2_enable:
            self.fc2 = Classifier(out_features, fc2_out_dim) 
            
    def encoder(self,x):
        x = self.f(x)
        out = torch.flatten(x, start_dim=1) 
        return out 
    
    def forward(self, x,return_encoding=False,return_projected_feature=False,classifier=False): 
        if return_projected_feature: 
            return self.projector(x)   
        if classifier:            
            if self.fc2_enable:
                return self.fc(x)  ,self.fc2(x)
            else:
                return self.fc(x) 
        feature=self.encoder(x)
        if return_encoding:          
            return feature
        out=self.fc(feature)
        if self.fc2_enable:
            out2=self.fc2(feature)
            return out,out2
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,fc2_enable=False,fc2_out_dim=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        out_features=512*block.expansion 
        self.projector=Projector(expansion=block.expansion)  
        self.fc2_enable=fc2_enable
        self.fc = nn.Linear(128, num_classes)
        assert not self.fc2_enable or self.fc2_enable and fc2_out_dim!=None
        if self.fc2_enable:
            self.fc2 = nn.Linear(128, fc2_out_dim) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def encoder(self,x):
        out = x         
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out) 
        out = self.layer1(out)         
        out = self.layer2(out) 
        out = self.layer3(out)  
        out = self.layer4(out) 
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out=self.projector(out)
        return out
        

    def forward(self, x, lin=0, lout=5, internal_outputs=False,
                classifier=False,return_encoding=False):  
        if classifier:            
            if self.fc2_enable:
                return self.fc(x)  ,self.fc2(x)
            else:
                return self.fc(x) 
        feature=self.encoder(x)
        if return_encoding:          
            return feature
        out=self.fc(feature)
        if self.fc2_enable:
            out2=self.fc2(feature)
            return out,out2
        return out

def Resnet20(cfg):
    num_classes=cfg.DATASET.NUM_CLASSES 
    use_norm = True if cfg.MODEL.LOSS.LABELED_LOSS == 'LDAM' else False
    return ResNet_s(BasicBlock, [3, 3, 3], num_classes=num_classes, use_norm=use_norm)


def Resnet32(cfg): 
    
    num_classes=cfg.DATASET.NUM_CLASSES 
    use_norm = True if cfg.MODEL.LOSS.LABELED_LOSS == 'LDAM' else False
    feature_dim=cfg.ALGORITHM.PRE_TRAIN.SimCLR.FEATURE_DIM 
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)

 
def Resnet44(cfg):
    
    num_classes=cfg.DATASET.NUM_CLASSES 
    use_norm = True if cfg.MODEL.LOSS.LABELED_LOSS == 'LDAM' else False
    return ResNet_s(BasicBlock, [7, 7, 7], num_classes=num_classes, use_norm=use_norm)


def Resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])

def Resnet34(cfg):    
    num_classes=cfg.DATASET.NUM_CLASSES  
    feature_dim=cfg.ALGORITHM.PRE_TRAIN.SimCLR.FEATURE_DIM
    fc2_enable=cfg.MODEL.DUAL_HEAD_ENABLE
    fc2_out_dim=cfg.MODEL.DUAL_HEAD_OUT_DIM    
    if fc2_out_dim==0:fc2_out_dim=None
    return Resnet34_(num_classes=num_classes,feature_dim=feature_dim,fc2_enable=fc2_enable,fc2_out_dim=fc2_out_dim)

def Resnet50(cfg): 
    num_classes=cfg.DATASET.NUM_CLASSES  
    feature_dim=cfg.ALGORITHM.PRE_TRAIN.SimCLR.FEATURE_DIM
    fc2_enable=cfg.MODEL.DUAL_HEAD_ENABLE
    fc2_out_dim=cfg.MODEL.DUAL_HEAD_OUT_DIM    
    if fc2_out_dim==0:fc2_out_dim=None 
    return ResNet(Bottleneck, [3,4,6,3],num_classes=num_classes,fc2_enable=fc2_enable,fc2_out_dim=fc2_out_dim)
 
def Resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def Resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))

 