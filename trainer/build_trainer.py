 
 
from .mood_trainer import MOODTrainer 
def build_trainer(cfg):
    alg=cfg.ALGORITHM.NAME
    if alg=='MOOD':
        return MOODTrainer(cfg)
    else:
        raise "The algorithm type is not valid!"