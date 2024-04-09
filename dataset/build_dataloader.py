

from dataset.build_dataset import *
import torch.utils.data as data
from dataset.build_sampler import *
import torch
from .random_sampler import RandomSampler
from .class_reversed_sampler import ClassReversedSampler
def build_contra_dataloader(args=None,cfg=None):
    
    train_dst = build_contra_dataset(cfg)
    train_sampler = None
    batch_size=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS
    train_loader = torch.utils.data.DataLoader(
        train_dst,batch_size=batch_size,num_workers=num_workers,
            pin_memory=True,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )
        
    return train_loader

def build_new_labeled_loader(cfg, new_l_dataset,sampler_type=None):
    batch_size=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS
    if sampler_type is not None:
        total_samples=len(new_l_dataset)
        sampler=build_sampler(cfg,new_l_dataset,sampler_type=sampler_type,total_samples=total_samples)
        new_labeled_trainloader=data.DataLoader(new_l_dataset,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                num_workers=num_workers,
                                                sampler=sampler)        
    else:
        new_labeled_trainloader=data.DataLoader(new_l_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return new_labeled_trainloader

def build_test_dataloader(cfg):
    test_set=build_test_dataset(cfg)
    batch_size=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS 
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
 
def build_domain_dataloader(cfg):
    test_domain_set= build_domain_dataset_for_ood_detection(cfg)  
    batch_size=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS
    domain_testloader = data.DataLoader(test_domain_set, batch_size=batch_size, shuffle=True, num_workers=num_workers) # 全部训练数据 DL DU OOD
    
    return domain_testloader

def build_warmup_loader(l_train,l_trans,sampler_name="RandomSampler",has_label=False,cfg=None,total_samples=0,
                        total_iter=0,
        drop_last=False): 
    batch_size = cfg.DATASET.BATCH_SIZE
    if not has_label:
        batch_size = int(batch_size * cfg.DATASET.DU.UNLABELED_BATCH_RATIO)
    if not has_label :
        sampler_name="RandomSampler"
    total_samples=batch_size*total_iter
    l_train = CIFAR10Dataset(l_train, transforms=l_trans)
    logger = logging.getLogger()
    if sampler_name == "RandomSampler":
        sampler = RandomSampler(l_train, total_samples)
        logger.info("Dual random labeled sampling branch is enabled! ")
    elif sampler_name == "ReversedSampler": 
        sampler = ClassReversedSampler(l_train, total_samples, shuffle=True)
        logger.info(
            "Dual ReversedSampler labeled sampling branchlabeled sampling branch is enabled.  "
            "per_class probabilities: {}".format(
                ", ".join(["{:.4f}".format(v) for v in sampler.per_cls_prob])
            )
        )   
    elif sampler_name == "ClassAwareSampler":
        beta = cfg.DATASET.SAMPLER_BETA
        sampler = ClassAwareSampler(l_train, total_samples, beta=beta, shuffle=True)
        logger.info(
            "Dual ClassAwareSampler labeled sampling branchlabeled sampling branch is enabled.  "
            "per_class probabilities: {}".format(
                ", ".join(["{:.4f}".format(v) for v in sampler.per_cls_prob])
            )
        )
        
    else:
        raise ValueError
    
    loader = data.DataLoader(
        l_train,
        batch_size=batch_size,
        num_workers=cfg.DATASET.NUM_WORKERS,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=False
    )
    return loader 


def build_data_loaders(cfg):
    builder = cfg.DATASET.BUILDER 
    l_train, ul_train, val_dataset, test_dataset = eval(builder)(cfg)
    l_loader = _build_loader(cfg, l_train) 
    ul_loader = None
    if ul_train is not None:
        ul_loader = _build_loader(cfg, ul_train, has_label=False)
    val_loader = None
    if val_dataset is not None:
        val_loader = _build_loader(cfg, val_dataset, is_train=False)
    test_loader = _build_loader(cfg, test_dataset, is_train=False) 
    return l_loader, ul_loader, val_loader, test_loader

def _build_loader(
    cfg, dataset, *, is_train = True,
    has_label= True,sampler_name="RandomSampler",total_samples=None
    
) : 
    sampler = None
    drop_last = is_train
    batch_size = cfg.DATASET.BATCH_SIZE

    if not has_label:
        batch_size = int(batch_size * cfg.DATASET.DU.UNLABELED_BATCH_RATIO)

    if is_train: 
        if not has_label:
            sampler_name = "RandomSampler"

        max_iter = cfg.MAX_EPOCH*cfg.TRAIN_STEP
        if total_samples is None:
            total_samples = max_iter * batch_size
        sampler=build_sampler(cfg, dataset,total_samples=total_samples,sampler_type=sampler_name)

    if (not has_label) and is_train and (cfg.ALGORITHM.NAME == "DARP_ESTIM"):
        sampler = None

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.DATASET.NUM_WORKERS,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=False
    )
    return data_loader

def build_dataloader(cfg,logger=None,test_mode=False):
    batch_size=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS
     
    datasets= build_dataset(cfg,logger,test_mode=test_mode)
    train_labeled_set=datasets[0]
    train_unlabeled_set=datasets[1] 
    train_domain_set=datasets[2]
    val_set=datasets[3]
    test_set=datasets[4] 
    pre_train_set=datasets[5]
    pre_train_loader=data.DataLoader(
        pre_train_set, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,drop_last=False)
    
    # ===  build dataloader  ===
    labeled_trainloader = _build_loader(cfg, train_labeled_set ) 
    
    unlabeled_trainloader = None
    if train_unlabeled_set is not None:
        unlabeled_trainloader = _build_loader(cfg, train_unlabeled_set, has_label=False,total_samples=len(train_unlabeled_set))
    val_loader = None
    if val_set is not None:
        val_loader = _build_loader(cfg, val_set, is_train=False)
    test_loader = _build_loader(cfg, test_set, is_train=False) 
    
    domain_trainloader=None
    if train_domain_set is not None:
        domain_trainloader= _build_loader(cfg, train_domain_set, has_label=False,total_samples=len(train_domain_set))

    return  domain_trainloader,labeled_trainloader,unlabeled_trainloader,val_loader,test_loader,pre_train_loader
    

def build_mood_dataloader(args,cfg):
    batch_size=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS
    datasets= build_dataset(cfg)
    train_labeled_set=datasets[0]
    train_unlabeled_set=datasets[1] 
    val_set=datasets[3]
    test_set=datasets[4]
    
    labeled_sampler = torch.utils.data.distributed.DistributedSampler(
            train_labeled_set,
            num_replicas=args.ngpu,
            rank=args.local_rank,
            )
    labeled_trainloader = data.DataLoader(
        train_labeled_set, 
        batch_size=batch_size, 
        shuffle=(labeled_sampler is None), 
        sampler=labeled_sampler,
        num_workers=num_workers,
        drop_last=False)
    unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(
            train_unlabeled_set,
            num_replicas=args.ngpu,
            rank=args.local_rank,
            )
    unlabeled_trainloader = data.DataLoader(
        train_unlabeled_set, 
        batch_size=batch_size, 
        shuffle=(unlabeled_sampler is None), 
        sampler=unlabeled_sampler,
        num_workers=num_workers,drop_last=True)
    
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    dual_sampler,dual_labeled_trainloader=None,None
    
    sampler_type=cfg.DATASET.DUAL_SAMPLER.NAME
    total_samples=len(labeled_trainloader.dataset)
     
     
    return  labeled_trainloader,labeled_sampler,unlabeled_trainloader,unlabeled_sampler,val_loader,test_loader
     