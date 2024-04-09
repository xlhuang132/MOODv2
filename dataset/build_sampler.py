
from .random_sampler import *
from .class_reversed_sampler import *
 

def build_sampler(cfg,dataset,sampler_type="RandomSampler",total_samples=None):
    assert sampler_type!=None and total_samples!=None
    if sampler_type == "RandomSampler": 
        sampler = RandomSampler(dataset,total_samples=total_samples)         
    
    elif sampler_type == "ClassReversedSampler":  
        sampler = ClassReversedSampler(dataset, total_samples)
        print(
                "ClassReversedSampler is enabled.  " 
            )
    else:
        raise ValueError
    
    
    return sampler

def build_dist_sampler(cfg,dataset,sampler_type="RandomSampler",total_samples=None,args=None):
    assert sampler_type!=None and total_samples!=None
    if sampler_type == "RandomSampler": 
        sampler = RandomSampler(dataset,total_samples=total_samples) 
    
    elif sampler_type == "ClassReversedSampler": 
        sampler = DistClassReversedSampler(dataset, total_samples,num_replicas=args.ngpu,
                                    rank=args.local_rank)
        print(
                "ClassReversedSampler is enabled.  " 
            )
    else:
        raise ValueError
    
    
    return sampler