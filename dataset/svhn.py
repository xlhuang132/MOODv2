import logging
import torchvision
from yacs.config import CfgNode
from .build_transform import  build_simclr_transform
from .base import BaseNumpyDataset
from .transform import build_transforms
from .utils import make_imbalance, map_dataset, split_trainval, split_val_from_train, x_u_split,ood_inject
import numpy as np
 
def get_svhn(root, out_dataset, start_label=0,ood_ratio=0, 
                 transform_train=None, transform_val=None,test_mode=False,
                 transform_train_ul=None,
                 download=True,cfg=None,logger=None):
  
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    num_l_head=cfg.DATASET.DL.NUM_LABELED_HEAD
    imb_factor_l=cfg.DATASET.DL.IMB_FACTOR_L 
    num_ul_head=cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD 
    imb_factor_ul=cfg.DATASET.DU.ID.IMB_FACTOR_UL 
    num_valid = cfg.DATASET.NUM_VALID
    reverse_ul_dist = cfg.DATASET.REVERSE_UL_DISTRIBUTION
    
    num_classes = cfg.DATASET.NUM_CLASSES
    seed = cfg.SEED
    
    ood_r=cfg.DATASET.DU.OOD.RATIO if cfg.DATASET.DU.OOD.ENABLE else 0
    ood_dataset=cfg.DATASET.DU.OOD.DATASET
    ood_root=cfg.DATASET.DU.OOD.ROOT
 

    logger = logging.getLogger()
     
    
    base_data = torchvision.datasets.SVHN(root, split='train', download=True)
    l_train=map_dataset(base_data)
 
    svhn_test = map_dataset(torchvision.datasets.SVHN(root, split='test', download=True))

    # train - valid set split
    svhn_valid = None
    if num_valid > 0:
        l_train, svhn_valid = split_trainval(l_train, num_valid, seed=seed)

    # unlabeled sample generation unber SSL setting
    ul_train = None
    l_train, ul_train = x_u_split(l_train, num_l_head, num_ul_head, seed=seed )
    if algorithm == "Supervised":
        ul_train = None

    # whether to shuffle the class order
    class_inds = list(range(num_classes))

    # make synthetic imbalance for labeled set
    if imb_factor_l > 1:
        l_train, class_inds = make_imbalance(
            l_train, num_l_head, imb_factor_l, class_inds, seed=seed,is_dl=True
        )

     
    # make synthetic imbalance for unlabeled set
    if ul_train is not None and imb_factor_ul > 1:
        ul_train, class_inds = make_imbalance(
            ul_train,
            num_ul_head,
            imb_factor_ul,
            class_inds,
            reverse_ul_dist=reverse_ul_dist,
            seed=seed
        )
    l_train['images'] =np.transpose(l_train['images'],(0,2,3,1)) 
    ul_train['images'] =np.transpose(ul_train['images'],(0,2,3,1))
    if ood_r>0:
        ul_train=ood_inject(ul_train,ood_root,ood_r,ood_dataset)
    
    labeled_data_num=len(l_train['labels'])
    domain_labels=np.hstack((np.ones_like(l_train['labels'],dtype=np.float32),np.zeros_like(ul_train['labels'],dtype=np.float32)))
    
    total_train={'images':np.vstack((l_train['images'],ul_train['images'])),
                 'labels':np.hstack((l_train['labels'],ul_train['labels'])),
                 }
    if ul_train is not None:
        ul_train = svhnDataset(ul_train, transforms=transform_train_ul,num_classes=num_classes)

    l_train = svhnDataset(l_train, transforms=transform_train,
                          num_classes=num_classes,
                          dual_sample=cfg.DATASET.DUAL_SAMPLER.ENABLE,
                          dual_sample_type=cfg.DATASET.DUAL_SAMPLER.NAME)
    
    if svhn_valid is not None:
        svhn_valid = svhnDataset(svhn_valid, transforms=transform_val,num_classes=num_classes)
    svhn_test = svhnDataset(svhn_test, transforms=transform_val,num_classes=num_classes)
    logger.info("class distribution of labeled dataset:{}".format(l_train.num_per_cls_list)) 
    logger.info(
        "=> number of labeled data: {}\n".format(
            sum( l_train.num_per_cls_list)
        )
    )
    if ul_train is not None:
        logger.info("class distribution of unlabeled dataset:{}".format(ul_train.num_per_cls_list)) 
        logger.info(
            "=> number of unlabeled ID data: {}\n".format(
                sum(ul_train.num_per_cls_list)
            )
        ) 
        logger.info(
            "=> number of unlabeled OOD data: {}\n".format( ul_train.ood_num)
        ) 
 
    train_dataset =svhnDataset(total_train,transforms=transform_train_ul,num_classes=num_classes)
    transform_pre=build_simclr_transform(cfg)
    pre_train_dataset  =  svhnDataset(total_train,transforms=transform_pre,num_classes=num_classes)
    return l_train, ul_train, train_dataset, svhn_valid, svhn_test,pre_train_dataset
 

class svhnDataset(BaseNumpyDataset):

    def __init__(self, *args, **kwargs):
        super(svhnDataset, self).__init__(*args, **kwargs)
