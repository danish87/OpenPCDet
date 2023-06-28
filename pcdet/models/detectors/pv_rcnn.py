from .detector3d_template import Detector3DTemplate
from ...utils.stats_utils import KITTIEvalMetrics, PredQualityMetrics
from torchmetrics.collections import MetricCollection
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import os
import torch
import pickle

class MetricRegistry(object):
    def __init__(self, **kwargs):
        self._tag_metrics = {}
        self.dataset = kwargs.get('dataset', None)
        self.cls_bg_thresh = kwargs.get('cls_bg_thresh', None)
        self.model_cfg = kwargs.get('model_cfg', None)
    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag in self._tag_metrics.keys():
            metric = self._tag_metrics[tag]
        else:
            kitti_eval_metric = KITTIEvalMetrics(tag=tag, dataset=self.dataset, config=self.model_cfg)
            pred_qual_metric = PredQualityMetrics(tag=tag, dataset=self.dataset, cls_bg_thresh=self.cls_bg_thresh, config=self.model_cfg)
            metric = MetricCollection({"kitti_eval_metric": kitti_eval_metric,
                                       "pred_quality_metric": pred_qual_metric})
            self._tag_metrics[tag] = metric
        return metric

    def tags(self):
        return self._tag_metrics.keys()
    
class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.metric_registry = MetricRegistry(dataset=self.dataset, model_cfg=model_cfg)
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'iteration','roi_labels', 'roi_scores']
        self.val_dict = {val: [] for val in vals_to_store}
        self._dict_map_ = {
                'iou_roi_pl': 'batch_iou_roi_pl[batch_index]',
                'iou_roi_gt': 'preds_iou_max',
                'pred_scores': 'batch_pred_score[batch_index]',
                'iteration': 'cur_iteration',
                'roi_labels': 'batch_roi_labels[batch_index]',
                'roi_scores': 'batch_roi_score[batch_index]'
            }
    def forward(self, batch_dict):
        batch_dict['metric_registry'] = self.metric_registry
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            if self.model_cfg.get('STORE_SCORES_IN_PKL', False) :
                # iter wise
                for tag_ in ['classwise_unlab_max_iou_bfs', 'classwise_unlab_max_iou_afs', 'classwise_lab_max_iou_bfs', 'classwise_lab_max_iou_afs']:
                    iter_name_ =  f'{tag_}_iteration'
                    if not (tag_ in self.roi_head.forward_ret_dict and self.roi_head.forward_ret_dict[tag_]): continue
                    cur_iteration = torch.ones_like(self.roi_head.forward_ret_dict[tag_][0]) * (batch_dict['cur_iteration'])
                    
                    if not iter_name_ in self.val_dict:
                        self.val_dict[iter_name_]=[]
                    self.val_dict[iter_name_].extend(cur_iteration.tolist())

                    for key, val in self.roi_head.forward_ret_dict[tag_].items():
                        name_ = f'{tag_}_{self.dataset.class_names[key]}'
                        if not name_ in self.val_dict:
                            self.val_dict[name_]=[]
                        self.val_dict[name_].extend(val.tolist())

                # batch wise
                batch_roi_labels = self.roi_head.forward_ret_dict['roi_labels'].detach().clone()
                batch_rois = self.roi_head.forward_ret_dict['rois'].detach().clone()
                batch_ori_gt_boxes = self.roi_head.forward_ret_dict['gt_boxes'].detach().clone()
                batch_iou_roi_pl = self.roi_head.forward_ret_dict['gt_iou_of_rois'].detach().clone()
                batch_pred_score = torch.sigmoid(batch_dict['batch_cls_preds']).detach().clone().squeeze()
                batch_roi_score =  torch.sigmoid(self.roi_head.forward_ret_dict['roi_scores']).detach().clone()

                for batch_index in range(len(batch_rois)):
                    valid_rois_mask = torch.logical_not(torch.all(batch_rois[batch_index] == 0, dim=-1))
                    valid_rois = batch_rois[batch_index][valid_rois_mask]
                    valid_roi_labels = batch_roi_labels[batch_index][valid_rois_mask]
                    valid_roi_labels -= 1  

                    valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[batch_index] == 0, dim=-1))
                    valid_gt_boxes = batch_ori_gt_boxes[batch_index][valid_gt_boxes_mask]
                    valid_gt_boxes[:, -1] -= 1  

                    num_gts = valid_gt_boxes_mask.sum()
                    num_preds = valid_rois_mask.sum()

                    if num_gts > 0 and num_preds > 0:
                        overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                        preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                        cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])

                        for val, map_val in self._dict_map_.items():
                            self.val_dict[val].extend(eval(map_val).tolist())
                
                # replace old pickle data (if exists) with updated one 
                #print([(k,len(v)) for k, v in self.val_dict.items() if len(v) ])
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'scores_lbd.pkl')
                pickle.dump(self.val_dict, open(file_path, 'wb'))

            for key in self.metric_registry.tags():
                metrics = self.compute_metrics(self.metric_registry, tag=key)
                tb_dict.update(metrics)

            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn

        # RCNN Entropy Regularization
        if self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("ENABLE_RCNN_ENTROPY_REG", False):
            lambda_=self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("RCNN_ENTROPY_REG_LAMBDA", 1.0)
            rcnn_entropy = lambda_ * self.roi_head.calc_entropy()
            tb_dict['rcnn_entropy'] = rcnn_entropy
            loss+= rcnn_entropy

        # RPN Entropy Regularization
        if self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("ENABLE_RPN_ENTROPY_REG", False):
            lambda_=self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("RPN_ENTROPY_REG_LAMBDA", 1.0)
            rpn_entropy = lambda_ * self.dense_head.calc_entropy()
            tb_dict['rpn_entropy'] = rpn_entropy
            loss+= rpn_entropy # overall reduction in the average uncertainty across all classes
        return loss, tb_dict, disp_dict

    def compute_metrics(self, registry, tag):
        results = registry.get(tag).compute()
        tag = tag + "/" if tag else ''
        metrics = {tag + key: val for key, val in results.items()}
        return metrics