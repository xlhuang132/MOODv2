
from .transforms import *
import torchvision
from .transform.rand_augment import RandAugment 
from .transform.transforms import SimCLRAugmentation
from dataset.transform.transforms import Augmentation,GeneralizedSSLTransform
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
 
def build_simclr_transform(cfg):
    dataset=cfg.DATASET.NAME
    
    resolution = cfg.DATASET.RESOLUTION
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset =='svhn':
        img_size = (32, 32)   
        norm_params = dict(mean=(0.4380, 0.4440, 0.4730), std=(0.1751, 0.1771, 0.1744))
    
    transform=SimCLRAugmentation(cfg, img_size,norm_params=norm_params, resolution=resolution)
    return TransformTwice(transform)

def build_transform(cfg):
    
    algo_name = cfg.ALGORITHM.NAME  
    
    resolution = cfg.DATASET.RESOLUTION
    
    dataset=cfg.DATASET.NAME
    aug = Augmentation 
     
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)
    elif dataset =='svhn':
        img_size = (32, 32)  # original image size
        norm_params = dict(mean=(0.4380, 0.4440, 0.4730), std=(0.1751, 0.1771, 0.1744))
    
    l_train = aug(
            cfg, img_size, 
            strong_aug=cfg.DATASET.TRANSFORM.LABELED_STRONG_AUG, 
            norm_params=norm_params, 
            resolution=resolution
        ) 
    
    if algo_name == "MixMatch":
        # K weak
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution)
                for _ in range(cfg.ALGORITHM.MIXMATCH.NUM_AUG)
            ]
        )
 
    elif algo_name=='OpenMatch':
        l_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                ),  # strong (randaugment)
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
            ]
        )
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                ),  # strong (randaugment)
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
            ]
        )
    else:
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),
                aug(
                    cfg,
                    img_size,
                    strong_aug=cfg.DATASET.TRANSFORM.UNLABELED_STRONG_AUG,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                )
            ]
        )
    eval_aug = Augmentation(
        cfg,
        img_size,
        flip=False,
        crop=False,
        norm_params=norm_params,
        is_train=False,
        resolution=resolution
    )
    return l_train,ul_train,eval_aug