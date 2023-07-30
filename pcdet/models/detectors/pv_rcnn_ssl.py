import os
import pickle
import torch
import copy

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from.pv_rcnn import PVRCNN

def _mean(tensor_list):
    tensor = torch.cat(tensor_list)
    tensor = tensor[~torch.isnan(tensor)]
    mean = tensor.mean() if len(tensor) > 0 else torch.tensor([float('nan')])
    return mean

class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        #self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.max_unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.unlabeled_weight_ascent_list = model_cfg.UNSUPERVISE_ASCENT_LIST
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        self._dict_map_ = {
                'iou_roi_pl': 'batch_iou_roi_pl[batch_index]',
                'iou_roi_gt': 'preds_iou_max',
                'angle_diff': 'batch_angle_diff[batch_index]',
                'center_dist': 'batch_center_dist[batch_index]',
                'iteration': 'cur_iteration',
                'roi_labels': 'batch_roi_labels[batch_index]',
                'roi_scores': 'batch_roi_score[batch_index]',
                'raw_roi_scores': 'batch_raw_roi_score[batch_index]',
                'pred_scores': 'batch_pred_score[batch_index]',
                'raw_pred_scores': 'batch_raw_pred_score[batch_index]',
                'target_labels': 'batch_target_labels[batch_index]'
            }
        self.val_lbd_dict = {val: [] for val in self._dict_map_.keys()}
        self.val_unlbd_dict = {val: [] for val in self._dict_map_.keys()}

    def forward(self, batch_dict):
        if self.training:
            cur_iteration, cur_epoch, total_it_each_epoch = (batch_dict['cur_iteration'], 
                                                             batch_dict['cur_epoch'], 
                                                             batch_dict['total_it_each_epoch'])
            
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
            tb_dict_ = {}
            disp_dict = {}
            ori_gt_boxes = batch_dict['gt_boxes']
            
            if cur_epoch > self.unlabeled_weight_ascent_list[0]:
                batch_dict_ema = {}
                keys = list(batch_dict.keys())
                for k in keys:
                    if k + '_ema' in keys:
                        continue
                    if k.endswith('_ema'):
                        batch_dict_ema[k[:-4]] = batch_dict[k]
                    else:
                        batch_dict_ema[k] = batch_dict[k]

                with torch.no_grad():
                    # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                    for cur_module in self.pv_rcnn_ema.module_list:
                        try:
                            batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                        except:
                            batch_dict_ema = cur_module(batch_dict_ema)
                    pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema,
                                                                                no_recall_dict=True, override_thresh=0.0, no_nms=self.no_nms)

                    pseudo_boxes = []
                    pseudo_scores = []
                    pseudo_sem_scores = []
                    max_box_num = batch_dict['gt_boxes'].shape[1]
                    max_pseudo_box_num = 0
                    for ind in unlabeled_inds:
                        pseudo_score = pred_dicts[ind]['pred_scores']
                        pseudo_box = pred_dicts[ind]['pred_boxes']
                        pseudo_label = pred_dicts[ind]['pred_labels']
                        pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']

                        if len(pseudo_label) == 0:
                            pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                            pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                            pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                            continue


                        conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                            0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))

                        valid_inds = pseudo_score > conf_thresh.squeeze()

                        valid_inds = valid_inds * (pseudo_sem_score > self.sem_thresh[0])

                        pseudo_sem_score = pseudo_sem_score[valid_inds]
                        pseudo_box = pseudo_box[valid_inds]
                        pseudo_label = pseudo_label[valid_inds]
                        pseudo_score = pseudo_score[valid_inds]

                        # if len(valid_inds) > max_box_num:
                        #     _, inds = torch.sort(pseudo_score, descending=True)
                        #     inds = inds[:max_box_num]
                        #     pseudo_box = pseudo_box[inds]
                        #     pseudo_label = pseudo_label[inds]

                        pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                        pseudo_sem_scores.append(pseudo_sem_score)
                        pseudo_scores.append(pseudo_score)

                        if pseudo_box.shape[0] > max_pseudo_box_num:
                            max_pseudo_box_num = pseudo_box.shape[0]
                        # pseudo_scores.append(pseudo_score)
                        # pseudo_labels.append(pseudo_label)

                    max_box_num = batch_dict['gt_boxes'].shape[1]

                    # assert max_box_num >= max_pseudo_box_num
                    #ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_inds, ...]

                    if max_box_num >= max_pseudo_box_num:
                        for i, pseudo_box in enumerate(pseudo_boxes):
                            diff = max_box_num - pseudo_box.shape[0]
                            if diff > 0:
                                pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                            batch_dict['gt_boxes'][unlabeled_inds[i]] = pseudo_box
                    else:
                        ori_boxes = batch_dict['gt_boxes']
                        new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                                device=ori_boxes.device)
                        for i, inds in enumerate(labeled_inds):
                            diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                            new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                            new_boxes[inds] = new_box
                        for i, pseudo_box in enumerate(pseudo_boxes):

                            diff = max_pseudo_box_num - pseudo_box.shape[0]
                            if diff > 0:
                                pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                            new_boxes[unlabeled_inds[i]] = pseudo_box
                        batch_dict['gt_boxes'] = new_boxes

                    batch_dict['gt_boxes'][unlabeled_inds, ...] = random_flip_along_x_bbox(
                        batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['flip_x'][unlabeled_inds, ...]
                    )

                    batch_dict['gt_boxes'][unlabeled_inds, ...] = random_flip_along_y_bbox(
                        batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['flip_y'][unlabeled_inds, ...]
                    )

                    batch_dict['gt_boxes'][unlabeled_inds, ...] = global_rotation_bbox(
                        batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['rot_angle'][unlabeled_inds, ...]
                    )

                    batch_dict['gt_boxes'][unlabeled_inds, ...] = global_scaling_bbox(
                        batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['scale'][unlabeled_inds, ...]
                    )

                    pseudo_ious = []
                    pseudo_accs = []
                    pseudo_fgs = []
                    sem_score_fgs = []
                    sem_score_bgs = []
                    for i, ind in enumerate(unlabeled_inds):
                        # statistics
                        anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                            batch_dict['gt_boxes'][ind, ...][:, 0:7],
                            ori_gt_boxes[unlabeled_inds, ...][i, :, 0:7])
                        cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                        unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                        cls_pseudo = cls_pseudo[unzero_inds]
                        if len(unzero_inds) > 0:
                            iou_max, asgn = anchor_by_gt_overlap[unzero_inds, :].max(dim=1)
                            pseudo_ious.append(iou_max.mean().unsqueeze(dim=0))
                            acc = (ori_gt_boxes[unlabeled_inds, ...][i][:, 7].gather(dim=0, index=asgn) == cls_pseudo).float().mean()
                            pseudo_accs.append(acc.unsqueeze(0))
                            fg = (iou_max > 0.5).float().sum(dim=0, keepdim=True) / len(unzero_inds)

                            sem_score_fg = (pseudo_sem_scores[i][unzero_inds] * (iou_max > 0.5).float()).sum(dim=0, keepdim=True) \
                                        / torch.clamp((iou_max > 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                            sem_score_bg = (pseudo_sem_scores[i][unzero_inds] * (iou_max < 0.5).float()).sum(dim=0, keepdim=True) \
                                        / torch.clamp((iou_max < 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                            pseudo_fgs.append(fg)
                            sem_score_fgs.append(sem_score_fg)
                            sem_score_bgs.append(sem_score_bg)

                            # only for 100% label
                            if self.supervise_mode >= 1:
                                filter = iou_max > 0.3
                                asgn = asgn[filter]
                                batch_dict['gt_boxes'][ind, ...][:] = torch.zeros_like(batch_dict['gt_boxes'][ind, ...][:])
                                batch_dict['gt_boxes'][ind, ...][:len(asgn)] = ori_gt_boxes[unlabeled_inds, ...][i, :].gather(dim=0, index=asgn.unsqueeze(-1).repeat(1, 8))

                                if self.supervise_mode == 2:
                                    batch_dict['gt_boxes'][ind, ...][:len(asgn), 0:3] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                        batch_dict['gt_boxes'][ind, ...][
                                                                                        :len(asgn), 3:6]
                                    batch_dict['gt_boxes'][ind, ...][:len(asgn), 3:6] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                        batch_dict['gt_boxes'][ind, ...][
                                                                                        :len(asgn), 3:6]
                        else:
                            nan = torch.tensor([float('nan')], device=unlabeled_inds.device)
                            sem_score_fgs.append(nan)
                            sem_score_bgs.append(nan)
                            pseudo_ious.append(nan)
                            pseudo_accs.append(nan)
                            pseudo_fgs.append(nan)
                
                tb_dict_['pseudo_ious'] = _mean(pseudo_ious)
                tb_dict_['pseudo_accs'] = _mean(pseudo_accs)
                tb_dict_['sem_score_fg'] = _mean(sem_score_fgs)
                tb_dict_['sem_score_bg'] = _mean(sem_score_bgs)
                tb_dict_['max_box_num'] = max_box_num
                tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

            
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            unlabeled_weight = self.calc_smooth_unlabeled_weight(cur_epoch)
            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].sum() + loss_rpn_cls[unlabeled_inds, ...].sum() * unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_inds, ...].sum() + loss_rpn_box[unlabeled_inds, ...].sum() * unlabeled_weight
            loss_point = loss_point[labeled_inds, ...].sum() + loss_point[unlabeled_inds, ...].sum() * 0.0 # sup comp minimization
            loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].sum() + loss_rcnn_cls[unlabeled_inds, ...].sum() * 0.0 # sup comp minimization

            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].sum()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].sum() + loss_rcnn_box[unlabeled_inds, ...].sum() * unlabeled_weight

            loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].sum()
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_inds, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]
            tb_dict_['unlabeled_weight']=unlabeled_weight
            ret_dict = {
                'loss': loss
            }

            if self.model_cfg.get('STORE_SCORES_IN_PKL', False) :
                # iter wise
                for tag_ in ['classwise_unlab_max_iou_bfs', 'classwise_unlab_max_iou_afs', 'classwise_lab_max_iou_bfs', 'classwise_lab_max_iou_afs']:
                    iter_name_ =  f'{tag_}_iteration'
                    if not (tag_ in self.pv_rcnn.roi_head.forward_ret_dict and self.pv_rcnn.roi_head.forward_ret_dict[tag_]): continue
                    cur_iteration = torch.ones_like(self.pv_rcnn.roi_head.forward_ret_dict[tag_][0]) * (batch_dict['cur_iteration'])
                    if 'unlab' in tag_:
                        if not iter_name_ in self.val_unlbd_dict:
                            self.val_unlbd_dict[iter_name_]=[]
                        self.val_unlbd_dict[iter_name_].extend(cur_iteration.tolist())
                    else:
                        if not iter_name_ in self.val_lbd_dict:
                            self.val_lbd_dict[iter_name_]=[]
                        self.val_lbd_dict[iter_name_].extend(cur_iteration.tolist())

                    for key, val in self.pv_rcnn.roi_head.forward_ret_dict[tag_].items():
                        name_ = f'{tag_}_{self.dataset.class_names[key]}'
                        if 'unlab' in tag_:
                            if not name_ in self.val_unlbd_dict:
                                self.val_unlbd_dict[name_]=[]
                            self.val_unlbd_dict[name_].extend(val.tolist())
                        else:
                            if not name_ in self.val_lbd_dict:
                                self.val_lbd_dict[name_]=[]
                            self.val_lbd_dict[name_].extend(val.tolist())
                
                # batch wise
                batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'].detach().clone()
                batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'].detach().clone()
                batch_ori_gt_boxes = self.pv_rcnn.roi_head.forward_ret_dict['ori_gt_boxes'].detach().clone() if 'ori_gt_boxes' in  self.pv_rcnn.roi_head.forward_ret_dict else ori_gt_boxes
                batch_iou_roi_pl = self.pv_rcnn.roi_head.forward_ret_dict['gt_iou_of_rois'].detach().clone()
                
                batch_center_dist = self.pv_rcnn.roi_head.forward_ret_dict['center_dist'].detach().clone()
                batch_angle_diff = self.pv_rcnn.roi_head.forward_ret_dict['angle_diff'].detach().clone()

                batch_raw_roi_score =  self.pv_rcnn.roi_head.forward_ret_dict['roi_scores'].detach().clone()
                batch_roi_score =  torch.sigmoid(batch_raw_roi_score)
                
                
                batch_raw_pred_score = batch_dict['batch_cls_preds'].detach().clone().squeeze()
                batch_pred_score = torch.sigmoid(batch_raw_pred_score)
                batch_target_labels = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'].detach().clone()


                
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
                        if batch_index in unlabeled_inds:
                            for val, map_val in self._dict_map_.items():
                                self.val_unlbd_dict[val].extend(eval(map_val).tolist())
                        else:
                            for val, map_val in self._dict_map_.items():
                                self.val_lbd_dict[val].extend(eval(map_val).tolist())
                
                # print([(k,len(v)) for k, v in self.val_lbd_dict.items() if len(v) ])
                # print([(k,len(v)) for k, v in self.val_unlbd_dict.items() if len(v) ])
                
                # replace old pickle data (if exists) with updated one 
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                pickle.dump(self.val_lbd_dict, open(os.path.join(output_dir, 'scores_lbd.pkl'), 'wb'))
                pickle.dump(self.val_unlbd_dict, open(os.path.join(output_dir, 'scores_unlbd.pkl'), 'wb'))
            

            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = self.model_cfg.EMA_ALPHA
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def calc_unlabeled_weight(self, cur_epoch):
        ascent_list = self.unlabeled_weight_ascent_list
        max_unlabeled_weight = self.max_unlabeled_weight

        if cur_epoch < ascent_list[0]:
            return 0.0
        elif cur_epoch >= ascent_list[-1]:
            return max_unlabeled_weight
        else:
            for i in range(1, len(ascent_list)):
                if cur_epoch < ascent_list[i]:
                    alpha = (cur_epoch - ascent_list[i - 1]) / (ascent_list[i] - ascent_list[i - 1])
                    return max_unlabeled_weight * (i - 1 + alpha) / len(ascent_list)

    def calc_smooth_unlabeled_weight(self,cur_epoch):
        ascent_list = self.unlabeled_weight_ascent_list
        max_unlabeled_weight = self.max_unlabeled_weight
        min_unlabeled_weight = 0.0
        if cur_epoch < ascent_list[0]:
            return min_unlabeled_weight
        elif cur_epoch >= ascent_list[-1]:
            return max_unlabeled_weight
        else:
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            alpha = (cur_epoch - ascent_list[0]) / (ascent_list[-1] - ascent_list[0])
            scaled_alpha = sigmoid(6 * (2 * alpha - 1))
            return min_unlabeled_weight + scaled_alpha * (max_unlabeled_weight - min_unlabeled_weight)
