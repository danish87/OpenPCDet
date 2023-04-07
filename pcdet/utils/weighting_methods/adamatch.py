import torch
from torchmetrics import Metric

class AdaMatchThreshold(Metric):
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



    def update(self, batch_label: torch.Tensor, batch_score: torch.Tensor) -> None:
        # Unsqueeze for DDP
        if batch_label.ndim == 1: batch_label=batch_label.unsqueeze(dim=0)
        if batch_score.ndim == 1:  batch_score=batch_score.unsqueeze(dim=0)
        self.scores.append(batch_score)
        self.labels.append(batch_label)


    def compute(self):
        all_score = [i.detach().cpu() for i in self.scores[-1]]    
        all_label = [i.detach().cpu() for i in self.labels[-1]]
        all_score = torch.cat(all_score, dim=0)
        all_label = torch.cat(all_label, dim=0)
        all_label -= 1

        glob_thr = torch.tensor(1.0 / self.num_classes)
        loc_thr = torch.ones((self.num_classes)) / self.num_classes
        for cind in range(self.num_classes):
            class_mask = all_label == cind
            class_score = all_score[class_mask]
            class_score = class_score[class_score>0]
            if class_score.shape[0]>0:
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
    



    





