import torch
from torchmetrics import Metric

"""
FreeMatch based Thresholding (Local and Global)
"""
class FreeMatchThreshold(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.momentum= kwargs.get('momentum', 0.99)
        self.quantile= kwargs.get('quantile', False)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3

        self.add_state("st_unlab_scores", default=[], dist_reduce_fx='cat')
        self.add_state("st_unlab_labels", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_scores", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_labels", default=[], dist_reduce_fx='cat')
        
        self.weights = None
        self.global_thresholds = torch.tensor(1.0 / self.num_classes)
        self.local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        
        self.time_p = torch.tensor(1.0 / self.num_classes)
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.unlabel_hist_ema = torch.ones((self.num_classes)) / self.num_classes
        self.fairness_loss = None
        



    def update(self, unlab_cls_labels: torch.Tensor, unlab_cls_scores: torch.Tensor, st_unlab_cls_labels: None, st_unlab_cls_scores: None) -> None:
        # Unsqueeze for DDP
        if unlab_cls_labels.ndim == 1: unlab_cls_labels=unlab_cls_labels.unsqueeze(dim=0)
        if unlab_cls_scores.ndim == 1:  unlab_cls_scores=unlab_cls_scores.unsqueeze(dim=0)
        self.unlab_scores.append(unlab_cls_scores)
        self.unlab_labels.append(unlab_cls_labels)
        if st_unlab_cls_labels is not None:
            if st_unlab_cls_labels.ndim == 1: st_unlab_cls_labels=st_unlab_cls_labels.unsqueeze(dim=0)
            self.st_unlab_labels.append(st_unlab_cls_labels)
        if st_unlab_cls_scores is not None:
            if st_unlab_cls_scores.ndim == 1:  st_unlab_cls_scores=st_unlab_cls_scores.unsqueeze(dim=0)
            self.st_unlab_scores.append(st_unlab_cls_scores)
        

    def compute(self):

        unlab_cls_scores = [i.detach().cpu() for i in self.unlab_scores[-1]]    
        unlab_cls_labels = [i.detach().cpu() for i in self.unlab_labels[-1]]
        unlab_cls_scores = torch.cat(unlab_cls_scores, dim=0)
        unlab_cls_labels = torch.cat(unlab_cls_labels, dim=0)
        self.weights = torch.zeros_like(unlab_cls_scores)
        
        valid_mask = unlab_cls_scores>0
        unlab_cls_scores = unlab_cls_scores[valid_mask]
        unlab_cls_labels = unlab_cls_labels[valid_mask]
        unlab_cls_labels -= 1
        unlabel_hist = torch.bincount(unlab_cls_labels, minlength=self.num_classes)
        unlabel_hist /= unlabel_hist.sum()
        self.unlabel_hist_ema = self.momentum  * self.unlabel_hist_ema + (1 - self.momentum) * unlabel_hist

        max_un_score=[] # classwise max score
        for cind in range(self.num_classes):
            unlab_class_score = unlab_cls_scores[unlab_cls_labels == cind]
            self.p_model[cind] = self.momentum  * self.p_model[cind] + (1 - self.momentum) * (
                torch.quantile(unlab_class_score, 0.8) if self.quantile else torch.mean(unlab_class_score))
            max_un_score.append(torch.max(unlab_class_score))

        max_un_score= torch.stack(max_un_score)
        self.time_p = self.momentum  * self.time_p + (1 - self.momentum) * torch.mean(max_un_score)# time_p: mean of max-prob classwise 
        #  self-adaptive threshold τt(c)
        self.local_thresholds = self.time_p * self.p_model / torch.max(self.p_model)
        self.local_thresholds = torch.clip(self.local_thresholds, 0.1, 0.9)

        for cind in range(self.num_classes):
            class_mask = unlab_cls_labels == cind
            unlab_class_score = unlab_cls_scores[class_mask]
            self.weights[valid_mask][class_mask] = unlab_class_score.ge(self.local_thresholds[cind])

        #  self-adaptive fairness
        if len(self.st_unlab_scores)>0:
            st_unlab_cls_scores = [i.detach().cpu() for i in self.st_unlab_scores[-1]]    
            st_unlab_cls_labels = [i.detach().cpu() for i in self.st_unlab_labels[-1]]
            st_unlab_cls_scores = torch.cat(st_unlab_cls_scores, dim=0)
            st_unlab_cls_labels = torch.cat(st_unlab_cls_labels, dim=0)

            # valid mask is unlab_cls_scores>0
            st_unlab_cls_scores = st_unlab_cls_scores[valid_mask]
            st_unlab_cls_labels = st_unlab_cls_labels[valid_mask]
            st_unlab_cls_labels -= 1

            # here hist is made before aplying masking unlike freematch org implemntation
            st_unlabel_hist = torch.bincount(st_unlab_cls_labels, minlength=self.num_classes)
            st_unlabel_hist /= st_unlabel_hist.sum()
        
            mod_prob_model = self.p_model / self.unlabel_hist_ema
            mod_prob_model /= mod_prob_model.sum()

            for cind in range(self.num_classes):
                unlab_class_score = unlab_cls_scores[unlab_cls_labels == cind]
                mask = unlab_class_score.ge(self.local_thresholds[cind])
                st_unlab_class_score = st_unlab_cls_scores[st_unlab_cls_scores == cind]
                mod_mean_prob_s = torch.mean(st_unlab_class_score[mask])/st_unlabel_hist[cind]
                
            mod_mean_prob_s /= mod_mean_prob_s.sum()
            self.fairness_loss = torch.mean(mod_prob_model * torch.log(mod_mean_prob_s + 1e-12))





        

        self.reset()


    def get_value(self):
        self.compute()
        return self.global_thresholds, self.local_thresholds, self.weights, self.fairness_loss
    


