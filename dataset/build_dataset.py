from .cifar10 import * 
from .cifar100 import * 
from .svhn import *
from .build_transform import build_transform
from dataset.ood_dataset_map import ood_dataset_map

def build_test_dataset(cfg):
    dataset_name=cfg.DATASET.NAME
    root=cfg.DATASET.ROOT
    _,_,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        test_dataset=get_cifar10_test_dataset(root,transform_val=transform_val)        
    else:
        raise "Dataset name {} is not valid!".format(dataset_name)
    print("Test data distribution:"+str(test_dataset.num_per_cls_list))
    return test_dataset 

def build_dataset(cfg,logger=None,test_mode=False):
    dataset_name=cfg.DATASET.NAME
    dataset_root=cfg.DATASET.ROOT 
    assert cfg.DATASET.DU.OOD.DATASET in ood_dataset_map.keys()
    ood_dataset=ood_dataset_map[cfg.DATASET.DU.OOD.DATASET] if cfg.DATASET.DU.OOD.ENABLE else 'None'
    
    ood_ratio=cfg.DATASET.DU.OOD.RATIO
    transform_train,transform_train_ul,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        datasets=get_cifar10(dataset_root,  ood_dataset,ood_ratio=ood_ratio, 
                 transform_train=transform_train,transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='cifar100':
        datasets=get_cifar100(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='svhn':
        datasets=get_svhn(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    else:
        raise "Dataset is not valid!"
    
    return datasets

def build_contra_dataset(cfg):
    dataset_name=cfg.DATASET.NAME
    dataset_root=cfg.DATASET.ROOT 
    assert cfg.DATASET.DU.OOD.DATASET in ood_dataset_map.keys()
    ood_dataset=ood_dataset_map[cfg.DATASET.DU.OOD.DATASET] if cfg.DATASET.DU.OOD.ENABLE else 'None'
    
    ood_ratio=cfg.DATASET.DU.OOD.RATIO
    transform_train,transform_train_ul,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        datasets=get_contra_cifar10(cfg)
    elif dataset_name=='cifar100':
        datasets=get_contra_cifar100(cfg)
    elif dataset_name=='svhn':
        datasets=get_svhn(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    else:
        raise "Dataset is not valid!"
    
    return datasets
        
    