import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torchmetrics import Metric
"""
Consistent-Teacher Adaptive Local Thresholding using GMM
"""
class AdaptiveThresholdGMM(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_clipping = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_CLIPPING', False)
        self.momentum= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MOMENTUM', 0.99)
        self.gmm_policy=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('GMM_POLICY','high')
        self.mu1=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MU1',0.1)
        self.mu2=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MU2',0.9)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3
        
        self.add_state("unlab_roi_labels", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_roi_scores", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_gt_iou_of_rois", default=[], dist_reduce_fx='cat') #iou_wrt_pl
        
        self.cls_local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.iou_local_thresholds = torch.ones((self.num_classes)) / self.num_classes

        self.gmm = GaussianMixture(
            n_components=2,
            weights_init=[0.5, 0.5],
            means_init=[[self.mu1], [self.mu2]],
            precisions_init=[[[1.0]], [[1.0]]],
            init_params='k-means++',
            tol=1e-9,
            max_iter=1000
        )
        

    def update(self, unlab_roi_labels: torch.Tensor,  
               unlab_roi_scores: torch.Tensor, 
               unlab_gt_iou_of_rois: torch.Tensor ) -> None:

        self.unlab_roi_labels.append(unlab_roi_labels)
        self.unlab_roi_scores.append(unlab_roi_scores)
        self.unlab_gt_iou_of_rois.append(unlab_gt_iou_of_rois)


    def compute(self):
        results = {}

        if  len(self.unlab_gt_iou_of_rois) >= self.reset_state_interval:

            unlab_roi_labels = [i.clone().detach() for i in self.unlab_roi_labels]
            unlab_roi_scores = [i.clone().detach() for i in self.unlab_roi_scores]    
            unlab_gt_iou_of_rois = [i.clone().detach() for i in self.unlab_gt_iou_of_rois]
            unlab_roi_labels = torch.cat(unlab_roi_labels, dim=0)
            unlab_roi_scores = torch.cat(unlab_roi_scores, dim=0)
            unlab_gt_iou_of_rois = torch.cat(unlab_gt_iou_of_rois, dim=0)
            unlab_roi_labels -= 1
            

            cls_loc_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  
            iou_loc_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  


            for cind in range(self.num_classes):
                
                cls_mask = unlab_roi_labels == cind
                cls_score = unlab_roi_scores[cls_mask]
                cls_score = cls_score[cls_score>self.pre_filtering_thresh].cpu().numpy()
                iou_score = unlab_gt_iou_of_rois[cls_mask]
                iou_score = iou_score[iou_score>self.pre_filtering_thresh].cpu().numpy()

                # cls
                if cls_score.shape[0]>4:
                    cls_score = cls_score.reshape(-1, 1)

                    self.gmm.fit(cls_score)  
                    gmm_assignment = self.gmm.predict(cls_score)  
                    gmm_scores = self.gmm.score_samples(cls_score) 
                    adaptive_thr=apply_policy_gmm(cls_score, gmm_assignment, gmm_scores, gmm_policy=self.gmm_policy)
                    if adaptive_thr is not None:
                        cls_loc_thr[cind] = adaptive_thr
                # iou
                if iou_score.shape[0]>4:
                    iou_score = iou_score.reshape(-1, 1)
                    self.gmm.fit(iou_score)  
                    gmm_assignment = self.gmm.predict(iou_score)  
                    gmm_scores = self.gmm.score_samples(iou_score) 
                    adaptive_thr=apply_policy_gmm(iou_score, gmm_assignment, gmm_scores, gmm_policy=self.gmm_policy)
                    if adaptive_thr is not None:
                        iou_loc_thr[cind] = adaptive_thr
                        
                    

            self.cls_local_thresholds = self.momentum  * self.cls_local_thresholds + (1 - self.momentum) * cls_loc_thr
            self.iou_local_thresholds = self.momentum  * self.iou_local_thresholds + (1 - self.momentum) * iou_loc_thr

            if self.enable_clipping:
                self.cls_local_thresholds = torch.clip(self.cls_local_thresholds, 0.1, 0.9)
                self.iou_local_thresholds = torch.clip(self.iou_local_thresholds, 0.1, 0.9)

            results.update(**{'cons_teacher_cls_local_thr': {cls_name: self.cls_local_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       'cons_teacher_iou_local_thr': {cls_name: self.iou_local_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       })
            
                
            self.reset()

        return results          


def apply_policy_gmm(scores, gmm_assignment, gmm_scores, gmm_policy='high'):
    adaptive_thr = None
    if np.any(gmm_assignment == 1):  
        if gmm_policy == 'high':
            gmm_scores[gmm_assignment == 0] = -np.inf  
            index = np.argmax(gmm_scores, axis=0) 
            pos_indx = ((gmm_assignment == 1) & (scores >= scores[index]).squeeze())  
            if np.sum(pos_indx):  adaptive_thr = np.min(scores[pos_indx])
        elif gmm_policy == 'middle': adaptive_thr = np.min(scores[gmm_assignment == 1])
        elif gmm_policy == 'percentile75': adaptive_thr = np.percentile(scores[gmm_assignment == 1], 75)
        elif gmm_policy == 'percentile25': adaptive_thr = np.percentile(scores[gmm_assignment == 1], 25)
        else:
            raise ValueError("Invalid policy. Policy can be 'high', 'middle', 'percentile75', or 'percentile25'.")
    return adaptive_thr

  