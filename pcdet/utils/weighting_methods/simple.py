
import numpy as np
import torch
from torchmetrics import Metric

"""
USAGE:
tg = SimpleThreshold(....)
tg.update({'batch_label':batch_label
            'batch_score': batch_score})
global_thresholds, local_thresholds, weights = tg.get_thresholds()
"""


"""
Simple EMA based Thresholding (Local and Global)
"""
class SimpleThreshold(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.momentum= kwargs.get('momentum', 0.99)
        self.quantile= kwargs.get('quantile', False)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3

        self.add_state("scores", default=[], dist_reduce_fx='cat')
        self.add_state("labels", default=[], dist_reduce_fx='cat')
        self.weights = None
        self.local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.global_thresholds = torch.tensor(1.0 / self.num_classes)



    def update(self, cls_labels: torch.Tensor, cls_scores: torch.Tensor) -> None:
        # Unsqueeze for DDP
        if cls_labels.ndim == 1: cls_labels=cls_labels.unsqueeze(dim=0)
        if cls_scores.ndim == 1:  cls_scores=cls_scores.unsqueeze(dim=0)
        self.scores.append(cls_scores)
        self.labels.append(cls_labels)


    def compute(self):
        cls_scores = [i.detach().cpu() for i in self.scores[-1]]    
        cls_labels = [i.detach().cpu() for i in self.labels[-1]]
        cls_scores = torch.cat(cls_scores, dim=0)
        cls_labels = torch.cat(cls_labels, dim=0)
        valid_mask = cls_scores>0
        cls_scores = cls_scores[valid_mask]
        cls_labels = cls_labels[valid_mask]
        cls_labels -= 1

        glob_thr = torch.tensor(1.0 / self.num_classes)
        loc_thr = torch.ones((self.num_classes)) / self.num_classes
        for cind in range(self.num_classes):
            class_mask = cls_labels == cind
            if class_mask.sum():
                class_score = cls_scores[class_mask]
                glob_thr += torch.max(class_score)
                loc_thr[cind] = torch.quantile(class_score, 0.8) if self.quantile else torch.mean(class_score)
        
        self.global_thresholds = self.momentum  * self.local_thresholds + (1 - self.momentum) * (glob_thr/self.num_classes)
        self.local_thresholds = self.momentum  * self.local_thresholds + (1 - self.momentum) * loc_thr
        if self.enable_clipping:
            self.global_thresholds = torch.clip(self.global_thresholds, 0.1, 0.9)
            self.local_thresholds = torch.clip(self.local_thresholds, 0.1, 0.9)
        
        self.reset()


    def get_value(self):
        self.compute()
        return self.global_thresholds, self.local_thresholds, self.weights
    



    





