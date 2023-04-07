from .detector3d_template import Detector3DTemplate
import pickle
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import os
import torch


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'iteration','roi_labels', 'roi_scores']
        self.val_dict = {val: [] for val in vals_to_store}

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            if self.model_cfg.get('STORE_SCORES_IN_PKL', False) :

                batch_roi_labels = self.roi_head.forward_ret_dict['roi_labels'].detach().clone()
                batch_rois = self.roi_head.forward_ret_dict['rois'].detach().clone()
                batch_ori_gt_boxes = self.roi_head.forward_ret_dict['gt_boxes'].detach().clone()
                batch_iou_roi_pl = self.roi_head.forward_ret_dict['gt_iou_of_rois'].detach().clone()
                batch_pred_score = torch.sigmoid(batch_dict['batch_cls_preds']).detach().clone().squeeze()
                batch_roi_score =  torch.sigmoid(self.roi_head.forward_ret_dict['roi_scores']).detach().clone()

                for i in range(len(batch_rois)):
                    valid_rois_mask = torch.logical_not(torch.all(batch_rois[i] == 0, dim=-1))
                    valid_rois = batch_rois[i][valid_rois_mask]
                    valid_roi_labels = batch_roi_labels[i][valid_rois_mask]
                    valid_roi_labels -= 1  

                    valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[i] == 0, dim=-1))
                    valid_gt_boxes = batch_ori_gt_boxes[i][valid_gt_boxes_mask]
                    valid_gt_boxes[:, -1] -= 1  

                    num_gts = valid_gt_boxes_mask.sum()
                    num_preds = valid_rois_mask.sum()

                    
                    if num_gts > 0 and num_preds > 0:
                        # Find IoU between ROI v/s Original GTs
                        overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                        preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                        cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])

                        self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())
                        self.val_dict['iou_roi_pl'].extend(batch_iou_roi_pl[i].tolist())
                        self.val_dict['pred_scores'].extend(batch_pred_score[i].tolist())
                        self.val_dict['roi_scores'].extend(batch_roi_score[i].tolist())
                        self.val_dict['roi_labels'].extend(batch_roi_labels[i].tolist())
                        self.val_dict['iteration'].extend(cur_iteration.tolist())
                        #print([len(v) for k, v in self.val_dict.items()])
                
                # replace old pickle data (if exists) with updated one 
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'scores.pkl')
                pickle.dump(self.val_dict, open(file_path, 'wb'))

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
        return loss, tb_dict, disp_dict
