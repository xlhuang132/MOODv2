
 
import torch  
import numpy as np 
import pandas as pd 
import os    
import torch.nn.functional as F
from utils.misc import AverageMeter  
from .base_trainer import BaseTrainer 
import math
from loss.contrastive_loss import *
from models.feature_queue import FeatureQueue
class MOODTrainer(BaseTrainer):
    def __init__(self, cfg):  
              
        super().__init__(cfg)      
           
        # =========== for ood detector ============== 
        self.cfg=cfg
        self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True) 
        self.feature_decay=0.9
        self.lambda_pap=cfg.ALGORITHM.MOOD.PAP_LOSS_WEIGHT
        self.feature_loss_temperature=cfg.ALGORITHM.MOOD.FEATURE_LOSS_TEMPERATURE
        self.alpha = cfg.ALGORITHM.MOOD.MIXUP_ALPHA       
        self.pap_loss_weight=cfg.ALGORITHM.MOOD.PAP_LOSS_WEIGHT
        self.loss_init()
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
         
        
    def train_step(self,pretraining=False):
        if self.pretraining:
            return self.train_warmup_step()
        else:
            return self.train_mood_step()
        
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_pap = AverageMeter() 
    
    def train_mood_step(self):
        self.model.train()
        loss_dict={}
        # DL  
        try: 
            inputs_x, targets_x,meta = self.labeled_train_iter.next()
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader) 
            inputs_x, targets_x,meta = self.labeled_train_iter.next() 
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        # DU  
        try:
            (inputs_u,inputs_u2),ul_y,u_index = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            (inputs_u,inputs_u2),ul_y,u_index = self.unlabeled_train_iter.next()
        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        ul_y=ul_y.cuda()            
        u_index=u_index.cuda()            
        id_mask,ood_mask = torch.ones_like(ul_y).cuda(),torch.zeros_like(ul_y).cuda()      
        inputs_dual_x, targets_dual_x=meta['dual_image'],meta['dual_label']  
        inputs_dual_x, targets_dual_x = inputs_dual_x.cuda(), targets_dual_x.cuda(non_blocking=True)        
        
        id_mask=self.id_masks[u_index].detach() 
        ood_mask=self.ood_masks[u_index].detach()   
        ood_index = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)  
        inputs_dual_x=self.mix_up(inputs_dual_x, inputs_u[ood_index]) 
        inputs_x=torch.cat([inputs_x,inputs_dual_x],0)
        targets_x=torch.cat([targets_x,targets_dual_x],0)
        l_encoding = self.model(inputs_x,return_encoding=True)  
        
         # 1. cls loss
        l_logits = self.model(l_encoding,classifier=True)    
        cls_loss = self.l_criterion(l_logits, targets_x)
        
        loss_dict.update({"loss_cls": cls_loss})
        
        score_result = self.func(l_logits)
        now_result = torch.argmax(score_result, 1)          

        # 2. cons loss
        ul_images=torch.cat([inputs_u , inputs_u2],0)
        ul_encoding=self.model(ul_images,return_encoding=True) 
        ul_logits = self.model(ul_encoding,classifier=True) 
        logits_weak, logits_strong = ul_logits.chunk(2)
        with torch.no_grad(): 
            p = logits_weak.detach().softmax(dim=1)   
            confidence, pred_class = torch.max(p, dim=1)

        loss_weight = confidence.ge(self.conf_thres).float()
        
        loss_weight*=id_mask
        cons_loss = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=logits_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})
          
        l_feature=self.model(l_encoding,return_projected_feature=True)
        l_feature = l_feature/(l_feature.norm(dim=1).view(l_feature.size(0),1) + 1e-8)
        with torch.no_grad():  
            self.queue.enqueue(l_feature.cpu().clone().detach(), targets_x.cpu().clone().detach())
             
        # 3. pap loss         
        ul_feature=self.model(ul_encoding,return_projected_feature=True)    
        ul_feature = ul_feature/(ul_feature.norm(dim=1).view(ul_feature.size(0),1) + 1e-8)              
        ul_feature_weak,ul_feature_strong=ul_feature.chunk(2)
        all_features=torch.cat((l_feature,ul_feature_weak),dim=0)
        all_target=torch.cat((targets_x,pred_class),dim=0)
        confidenced_id_mask= torch.cat([torch.ones(l_feature.size(0)).cuda(),id_mask*loss_weight],dim=0).long() 
        Lidfeat=  self.pap_loss_weight*self.get_id_feature_contrast_loss(all_features, all_target, confidenced_id_mask)
        Loodfeat=0.
        if ood_mask.sum()>0:        
            Loodfeat=self.pap_loss_weight*self.get_ood_feature_contrast_loss(ul_feature_weak,ul_feature_strong,ood_mask) 
        pap_loss=Lidfeat+Loodfeat    
        loss_dict.update({"pap_loss":pap_loss})  
        loss = sum(loss_dict.values())
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(cls_loss.item(), inputs_x.size(0))
        self.losses_u.update(cons_loss.item(), inputs_u.size(0)) 
        self.losses_pap.update(pap_loss.item(),inputs_x.size(0)+inputs_u.size(0))
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_loss:{:>5.4f} Loss_x:{:>5.4f}  Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.val,self.losses_x.avg,self.losses_u.val))
            self.logger.info('================= Loss_pap_id:{:>5.4f} Loss_pap_ood:{:>5.4f} ==============='.format(self.losses_pap_id.val,self.losses_pap_ood.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()  
   
    def get_id_feature_contrast_loss(self,feature,targets,id_mask):
        prototypes = self.queue.prototypes.cuda()
        outputs=torch.cat((feature,prototypes),dim=0)
        B   = outputs.shape[0]
        outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
        similarity_matrix = (1./self.feature_loss_temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))[:feature.size(0),feature.size(0):]
        mask_same_c=torch.eq(\
            targets.contiguous().view(-1, 1).cuda(), \
            torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
        id_mask=id_mask.expand(mask_same_c.size(1),-1).T  
        mask_same_c*=id_mask 
        similarity_matrix_exp = torch.exp(similarity_matrix)         
        log_prob = similarity_matrix_exp - torch.log((torch.exp(similarity_matrix_exp) * (1 - mask_same_c)).sum(1, keepdim=True))
        log_prob_pos = log_prob * mask_same_c  
        loss = - log_prob_pos.sum() / mask_same_c.sum()  
        return loss
    
    def get_ood_feature_contrast_loss(self,features_u,features_u2,ood_mask): 
        prototypes = self.queue.prototypes.cuda()
        features=torch.cat([features_u,features_u2],0) 
        all_features=torch.cat([features,prototypes],0)  
        B   = all_features.shape[0]
        outputs_norm = all_features/(all_features.norm(dim=1).view(B,1) + 1e-8)
        similarity_matrix = (1./self.feature_loss_temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))[:features_u.size(0),features_u.size(0):]
        mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),self.num_classes))],dim=1)
        mask_same_c=mask_same_c.cuda()
        ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
        mask_same_c*=ood_mask
        log_prob = similarity_matrix - torch.log((torch.exp(similarity_matrix) * (1 - mask_same_c)).sum(1, keepdim=True))
        log_prob_pos = log_prob * mask_same_c 
        loss = - log_prob_pos.sum() / mask_same_c.sum()    
        return loss   
    
    def mix_up(self,l_images,ul_images):                 
        with torch.no_grad():     
            len_l=l_images.size(0)
            len_aux=ul_images.size(0)
            if len_aux==0: 
                return l_images
            elif len_aux>len_l:
                ul_images=ul_images[:len_l]
            elif len_aux<len_l:
                extend_num= math.ceil(len_l/len_aux)
                tmp=[ul_images]*extend_num
                ul_images=torch.cat(tmp,dim=0)[:len_l]

            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1.0 - lam)
            rand_idx = torch.randperm(l_images.size(0)) 
            mixed_images = lam * l_images + (1.0 - lam) * ul_images[rand_idx]
            return mixed_images  
        
    def operate_after_epoch(self): 
        if self.iter<=self.warmup_iter:            
            self.detect_ood()
        if self.iter==self.warmup_iter:
            self.save_checkpoint(file_name="warmup_model.pth")
            # train from scratch
            self._rebuild_models()
            self._rebuild_optimizer(self.model) 
            torch.cuda.empty_cache()
         
    def _rebuild_models(self):
        model = self.build_model(self.cfg) 
        self.model = model.cuda()    
  
    
    def save_checkpoint(self,file_name=""):
        if file_name=="":
            file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
        torch.save({
                    'model': self.model.state_dict(),
                    'ema_model': self.ema_model.state_dict(),
                    'iter': self.iter, 
                    'best_val': self.best_val, 
                    'best_val_iter':self.best_val_iter, 
                    'best_val_test': self.best_val_test,
                    'optimizer': self.optimizer.state_dict(), 
                    "prototypes":self.queue.prototypes, 
                    'id_masks':self.id_masks,
                    'ood_masks':self.ood_masks,                   
                },  os.path.join(self.model_dir, file_name))
        return    
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")
        if not os.path.exists(resume):
            self.logger.info(f"Can\'t resume form {resume}")
            
        state_dict = torch.load(resume)
        model_dict=self.model.state_dict() 
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict["model"].items() if k in self.model.state_dict()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict) 
        self.optimizer.load_state_dict(state_dict["optimizer"])   
        
        self.id_masks=state_dict['id_masks']
        self.ood_masks=state_dict['ood_masks']
        
        finetune= not 'warmup' in self.cfg.RESUME 
        if not finetune:
            self._rebuild_models() 
            self._rebuild_optimizer(self.model) 
            self.logger.info('== Rebuild model and optimizer ==')
        else:
            self.logger.info('== Finetune model ==')
            
        self.queue.prototypes=state_dict["prototypes"] 
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.train_per_step)+1 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )
        
   