import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torchmetrics import Metric
from pcdet.ops.iou3d_nms import iou3d_nms_utils


"""
Consistent-Teacher Adaptive Local Thresholding using GMM
"""
class AdaptiveThresholdGMM(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RESET_STATE_INTERVAL', 32)
        self.PRE_FILTERING_THRESH=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_PLOTS', False)
        self.enable_clipping = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_CLIPPING', False)
        self.momentum= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MOMENTUM', 0.99)
        self.gmm_policy=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('GMM_POLICY','high')
        self.mu1=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MU1',0.1)
        self.mu2=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MU2',0.9)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3
        #self.min_overlaps = np.array([0.7, 0.5, 0.5, 0.7, 0.5, 0.7])
        self.fg_gt_thresh = [0.7, 0.5, 0.5]
        self.fg_pl_thresh = [0.65, 0.45, 0.4]
        self.bg_gt_thresh = 0.25
        #self.dataset = kwargs.get('dataset', None)
        #self.default_C =  torch.tensor([self.dataset.class_counter[cls_name] for _, cls_name in self.class_names.items()])
        #self.default_C = self.default_C/self.default_C.sum()
        
        
        self.add_state("batch_cls_score", default=[], dist_reduce_fx='cat')
        self.add_state("batch_iou_score", default=[], dist_reduce_fx='cat')
        self.add_state("batch_label", default=[], dist_reduce_fx='cat')
        self.add_state("batch_rois", default=[], dist_reduce_fx='cat')
        self.add_state("batch_gts", default=[], dist_reduce_fx='cat')
        self.add_state("lab_iou_label", default=[], dist_reduce_fx='cat')
        self.add_state("lab_iou_score", default=[], dist_reduce_fx='cat')

        
        
        self.cls_local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.iou_local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.iteration_count=0

        self.gmm = GaussianMixture(
            n_components=2,
            weights_init=[0.5, 0.5],
            means_init=[[self.mu1], [self.mu2]],
            precisions_init=[[[1.0]], [[1.0]]],
            init_params='k-means++',
            tol=1e-9,
            max_iter=1000
        )
        

    def update(self, batch_label: torch.Tensor, batch_cls_score: torch.Tensor, batch_iou_score: torch.Tensor, 
               batch_rois: torch.Tensor, batch_gts: torch.Tensor, 
               lab_iou_score: torch.Tensor, lab_iou_label: torch.Tensor) -> None:

        self.lab_iou_label.append(lab_iou_label)
        self.lab_iou_score.append(lab_iou_score)
        self.batch_gts.append(batch_gts)
        self.batch_rois.append(batch_rois)
        self.batch_iou_score.append(batch_iou_score)
        self.batch_cls_score.append(batch_cls_score)
        self.batch_label.append(batch_label)


    def compute(self):
        results = {}

        if  len(self.batch_iou_score) >= self.reset_state_interval:
            self.iteration_count+=1
            
            
            batch_cls_score = [i.clone().detach() for i in self.batch_cls_score]    
            batch_iou_score = [i.clone().detach() for i in self.batch_iou_score]
            batch_label = [i.clone().detach() for i in self.batch_label]
            batch_rois = [i.clone().detach() for i in self.batch_rois]
            batch_gts = [i.clone().detach() for i in self.batch_gts]
            lab_iou_score = [i.clone().detach() for i in self.lab_iou_score]
            lab_iou_label = [i.clone().detach() for i in self.lab_iou_label]

            batch_cls_score = torch.cat(batch_cls_score, dim=0)
            batch_iou_score = torch.cat(batch_iou_score, dim=0)
            batch_label = torch.cat(batch_label, dim=0)
            batch_rois = torch.cat(batch_rois, dim=0)
            batch_gts = torch.cat(batch_gts, dim=0)
            lab_iou_score = torch.cat(lab_iou_score, dim=0)
            lab_iou_label = torch.cat(lab_iou_label, dim=0)
            
            batch_label -= 1
            lab_iou_label -= 1
            

            cls_loc_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  
            iou_loc_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  
            cls_plot_handler = {cind:None for cind in range(self.num_classes)}
            iou_plot_handler = {cind:None for cind in range(self.num_classes)}

            for cind in range(self.num_classes):
                
                cls_mask = batch_label == cind
                cls_score = batch_cls_score[cls_mask]
                cls_score = cls_score[cls_score>self.PRE_FILTERING_THRESH].cpu().numpy()
                iou_score = batch_iou_score[cls_mask]
                iou_score = iou_score[iou_score>self.PRE_FILTERING_THRESH].cpu().numpy()
                info_=f"{self.class_names[cind]} Iter {(self.iteration_count-1)*self.reset_state_interval} : {self.iteration_count*self.reset_state_interval}\n"

                # cls
                if cls_score.shape[0]>4:
                    cls_score = cls_score.reshape(-1, 1)

                    self.gmm.fit(cls_score)  
                    gmm_assignment = self.gmm.predict(cls_score)  
                    gmm_scores = self.gmm.score_samples(cls_score) 
                    adaptive_thr=apply_policy_gmm(cls_score, gmm_assignment, gmm_scores, gmm_policy=self.gmm_policy)
                    if self.enable_plots:
                        cls_plot_handler[cind]=plot_gmm_modes(
                            cls_score, self.gmm, gmm_assignment, 
                            default_thr=cls_loc_thr[cind], 
                            adaptive_thr=adaptive_thr, 
                            ema_thr=self.cls_local_thresholds[cind], 
                            info=info_ + f" Classification Score GMM Modeling\nConverged: {self.gmm.converged_}\nLogLikelihood: {self.gmm.lower_bound_:.2f}\nGMM-Niter: {self.gmm.n_iter_}"
                        )
                    if adaptive_thr is not None:
                        cls_loc_thr[cind] = adaptive_thr
                # iou
                if iou_score.shape[0]>4:
                    iou_score = iou_score.reshape(-1, 1)
                    self.gmm.fit(iou_score)  
                    gmm_assignment = self.gmm.predict(iou_score)  
                    gmm_scores = self.gmm.score_samples(iou_score) 
                    adaptive_thr=apply_policy_gmm(iou_score, gmm_assignment, gmm_scores, gmm_policy=self.gmm_policy)
                    if self.enable_plots:
                        iou_plot_handler[cind]=plot_gmm_modes(
                            iou_score, self.gmm, gmm_assignment, 
                            default_thr=iou_loc_thr[cind], 
                            adaptive_thr=adaptive_thr, 
                            ema_thr=self.iou_local_thresholds[cind],
                            info=info_ + f" IoU Score GMM Modeling\nConverged: {self.gmm.converged_}\nLogLikelihood: {self.gmm.lower_bound_:.2f}\nGMM-Niter: {self.gmm.n_iter_}"
                        )
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
            if self.enable_plots:
                results.update(**{
                'cls_gmm_modes': {cls_name: cls_plot_handler[i] for i, cls_name in self.class_names.items() if cls_plot_handler[i] is not None},
                'iou_gmm_modes': {cls_name: iou_plot_handler[i] for i, cls_name in self.class_names.items() if iou_plot_handler[i] is not None}
                })
            
            
            if self.enable_plots:
                
                info = f"Iter {(self.iteration_count-1)*self.reset_state_interval} : {self.iteration_count*self.reset_state_interval}"
                bins = 20
                alpha = 0.6 
                
                #lab_unlab_dist_gmm
                fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
                for cind, class_name in self.class_names.items():
                   
                    cls_lab_iou_score = lab_iou_score[lab_iou_label==cind]
                    cls_unlab_iou_score = batch_iou_score[batch_label==cind]
                    cls_lab_iou_score = cls_lab_iou_score[cls_lab_iou_score>self.PRE_FILTERING_THRESH]
                    cls_unlab_iou_score = cls_unlab_iou_score[cls_unlab_iou_score>self.PRE_FILTERING_THRESH]

                    axs[cind].hist(cls_lab_iou_score.cpu().numpy(), bins=bins, edgecolor='black',label='Labeled Scores', alpha=alpha)
                    axs[cind].hist(cls_unlab_iou_score.cpu().numpy(), bins=bins, edgecolor='black',label='Unlabeled Scores', alpha=alpha)
                   
                    axs[cind].axvline(self.iou_local_thresholds[cind], color='r', linestyle='--', label='EMA (GMM)')
                    axs[cind].axvline(torch.tensor(1.0 / self.num_classes), color='c', linestyle='--', label='1/C')
                    axs[cind].set_xlabel('RoI IoU wrt PL', fontsize='x-small')
                    axs[cind].set_ylabel('Count', fontsize='x-small')
                    axs[cind].set_title(f'{class_name}', fontsize='x-small')
                    axs[cind].grid(True, alpha=0.2) 
                    
                    axs[cind].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='x-small')

                fig.suptitle(info, fontsize='medium')
                plt.tight_layout()
                results['lab_unlab_dist_gmm'] = fig.get_figure()
                plt.close()

                #cls_score_vs_iou_score
                fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)
                for cind, class_name in self.class_names.items():

                    cls_mask = batch_label == cind
                    cls_score = batch_cls_score[cls_mask]
                    iou_score = batch_iou_score[cls_mask]
                    cls_score = cls_score[cls_score>self.PRE_FILTERING_THRESH]
                    iou_score = iou_score[iou_score>self.PRE_FILTERING_THRESH]
                    if not cls_score.shape[0]==0:
                        axs[cind].hist(cls_score.cpu().numpy(),  bins=bins, edgecolor='black',label='cls-score', alpha=alpha)
                    if not iou_score.shape[0]==0:
                        axs[cind].hist(iou_score.cpu().numpy(),  bins=bins, edgecolor='black',label='iou-score', alpha=alpha)
                    
                    axs[cind].set_title(f'{class_name}', fontsize='x-small')
                    axs[cind].set_ylabel('Count', fontsize='x-small')
                    axs[cind].grid(True, alpha=0.2) 
                    axs[cind].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='x-small')
                    
                fig.suptitle(info, fontsize='medium')
                plt.tight_layout()
                results["cls_score_vs_iou_score"] = fig.get_figure()
                plt.close()
                
                
            self.reset()

        return results          


def apply_policy_gmm(scores, gmm_assignment, gmm_scores, gmm_policy='high'):
    adaptive_thr = None
    if not np.any(gmm_assignment == 1):  return adaptive_thr
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

  
def plot_gmm_modes(scores, gmm, gmm_assignment, default_thr=None, adaptive_thr=None, ema_thr=None, info=None):    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    _, bins = np.histogram(scores, bins=50)

    # Plot GMM components on the histogram
    gmm_scores = gmm.score_samples(bins.reshape(-1, 1))
    component_0_scores = scores[gmm_assignment == 0]
    component_1_scores = scores[gmm_assignment == 1]
    axs[0].hist(component_0_scores, bins=bins, color='b', alpha=0.5, label='Component 0')
    axs[0].hist(component_1_scores, bins=bins, color='r', alpha=0.5, label='Component 1')
    axs[0].set_xlabel('Scores',fontsize='x-small')
    axs[0].set_ylabel('Count',fontsize='x-small')
    axs[0].set_title('Histogram of GMM Components',fontsize='x-small')
    axs[0].legend(fontsize='x-small')
    axs[0].grid(True) 

    # Plot GMM PDF and mean values with decision boundaries
    gmm_x = np.linspace(np.min(scores), np.max(scores), 1000).reshape(-1, 1)
    gmm_scores = gmm.score_samples(gmm_x)
    axs[1].plot(gmm_x, np.exp(gmm_scores), color='k', label='GMM PDF')
    axs[1].axvline(x=gmm.means_[0], color='b', linestyle='--', label='mu 0')
    axs[1].axvline(x=gmm.means_[1], color='r', linestyle='--', label='mu 1')
    if adaptive_thr is not None:
        axs[1].axvline(x=adaptive_thr, color='m', linestyle='--', label='Ada')
    if default_thr is not None:
        axs[1].axvline(x=default_thr, color='c', linestyle='--', label='1/C')
    if ema_thr is not None:
        axs[1].axvline(x=ema_thr, color='y', linestyle='--', label='EMA')
    axs[1].set_xlabel('Scores',fontsize='x-small')
    axs[1].set_ylabel('Density',fontsize='x-small')
    axs[1].set_title('GMM PDF',fontsize='x-small')
    axs[1].legend(fontsize='x-small')
    axs[1].grid(True) 
    if info is not None:
        fig.suptitle(info, fontsize='small')
    fig.tight_layout()
    hh_ = fig.get_figure()
    plt.close()
    return hh_