import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        return self.sample_rois_for_rcnn(batch_dict=batch_dict)

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_reg_valid_mask = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_cls_labels = rois.new_ones(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE) #NOTE CHECK - SIGN HERE   
        interval_mask = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, dtype=torch.bool)
        # added for storing pkls
        classwise_lab_max_iou_bfs = {}
        classwise_lab_max_iou_afs = {}
        classwise_unlab_max_iou_bfs = {}
        classwise_unlab_max_iou_afs = {}
        
        # Sampling methods
        UNLABELED_SAMPLER = getattr(self, self.roi_sampler_cfg.UNLABELED_SAMPLER, None)
        LABELED_SAMPLER = getattr(self, self.roi_sampler_cfg.LABELED_SAMPLER, None)
        
        # Adaptive or Fixed threshold
        batch_dict['iou_fg_thresh'] = self.roi_sampler_cfg.UNLABELED_CLS_FG_THRESH 
        if 'thresh_registry' in batch_dict:
            if 'roi_iou_pl_adaptive_thresh_afs' in batch_dict['thresh_registry'].tags(): 
                    batch_dict['iou_fg_thresh'] = \
                        batch_dict['thresh_registry'].get(tag='roi_iou_pl_adaptive_thresh_afs').iou_local_thresholds.tolist()
        
        
        # main loop
        for index in range(batch_size):
            # TODO(farzad) WARNING!!! The index for cur_gt_boxes was missing and caused an error. FIX this in other branches.
            cur_gt_boxes = batch_dict['gt_boxes'][index] # 
            k = cur_gt_boxes.__len__() - 1
            while k >= 0 and cur_gt_boxes[k].sum() == 0:
                k -= 1
            cur_gt_boxes = cur_gt_boxes[:k + 1]
            cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(cur_gt_boxes) == 0 else cur_gt_boxes
            

            # choose a sampler 
            current_batch_sampler = self.default_class_agnostic_subsampler
            if index in batch_dict['unlabeled_inds'] and UNLABELED_SAMPLER is not None:
                current_batch_sampler = UNLABELED_SAMPLER
            elif index not in batch_dict['unlabeled_inds'] and LABELED_SAMPLER is not None:
                current_batch_sampler = LABELED_SAMPLER
                
            (sampled_inds, cur_reg_valid_mask, 
            cur_cls_labels, roi_ious, 
            gt_assignment, cur_interval_mask, 
            classwise_max_iou_bfs_batch, classwise_max_iou_afs_batch) = current_batch_sampler(batch_dict, index)

            
            batch_rois[index] = batch_dict['rois'][index][sampled_inds]
            batch_roi_labels[index] = batch_dict['roi_labels'][index][sampled_inds]
            batch_roi_scores[index] = batch_dict['roi_scores'][index][sampled_inds]
            batch_reg_valid_mask[index] = cur_reg_valid_mask
            batch_cls_labels[index] = cur_cls_labels
            batch_roi_ious[index] = roi_ious
            batch_gt_of_rois[index] = cur_gt_boxes[gt_assignment[sampled_inds]]
            interval_mask[index] = cur_interval_mask
            
            # ------------------- Temprorialy added for record keeping before and after subsampling --------------------
            for key, val in classwise_max_iou_bfs_batch.items():
                if index in batch_dict['unlabeled_inds']:
                    if key not in classwise_unlab_max_iou_bfs:
                        classwise_unlab_max_iou_bfs[key] = []
                    classwise_unlab_max_iou_bfs[key].extend(val)
                else:
                    if key not in classwise_lab_max_iou_bfs:
                        classwise_lab_max_iou_bfs[key] = []
                    classwise_lab_max_iou_bfs[key].extend(val)

            for key, val in classwise_max_iou_afs_batch.items():
                if index in batch_dict['unlabeled_inds']:
                    if key not in classwise_unlab_max_iou_afs:
                        classwise_unlab_max_iou_afs[key] = []
                    classwise_unlab_max_iou_afs[key].extend(val)
                else:
                    if key not in classwise_lab_max_iou_afs:
                        classwise_lab_max_iou_afs[key] = []
                    classwise_lab_max_iou_afs[key].extend(val)
            

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': batch_reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels,
                        'interval_mask': interval_mask,
                        'classwise_unlab_max_iou_bfs': {k: torch.stack(v) for k, v in classwise_unlab_max_iou_bfs.items()}, 
                        'classwise_unlab_max_iou_afs': {k: torch.stack(v) for k, v in classwise_unlab_max_iou_afs.items()},
                        'classwise_lab_max_iou_bfs': {k: torch.stack(v) for k, v in classwise_lab_max_iou_bfs.items()}, 
                        'classwise_lab_max_iou_afs': {k: torch.stack(v) for k, v in classwise_lab_max_iou_afs.items()}}
        
        return targets_dict
    
    def default_class_agnostic_subsampler(self, batch_dict, index):
        
        cur_roi = batch_dict['rois'][index]
        cur_gt_boxes = batch_dict['gt_boxes'][index]
        cur_roi_labels = batch_dict['roi_labels'][index]
        classwise_max_iou_bfs = {}
        classwise_max_iou_afs = {}

        k = cur_gt_boxes.__len__() - 1
        while k >= 0 and cur_gt_boxes[k].sum() == 0:
            k -= 1
        cur_gt_boxes = cur_gt_boxes[:k + 1]
        cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(
            cur_gt_boxes) == 0 else cur_gt_boxes

        if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=cur_roi, roi_labels=cur_roi_labels,
                gt_boxes=cur_gt_boxes[:, 0:7], gt_labels=cur_gt_boxes[:, -1].long())
                
        else:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt_boxes[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

        fg_inds, bg_inds = self.subsample_rois(max_overlaps=max_overlaps)
        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
            

        roi_ious = max_overlaps[sampled_inds]
        
        # regression valid mask
        reg_valid_mask = (roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()
        # classification label
        iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
        iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH

        fg_mask = roi_ious > iou_fg_thresh
        bg_mask = roi_ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        cls_labels = (fg_mask > 0).float()

        cls_labels[interval_mask] = \
            (roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        
        # ------------------- Temprorialy added for record keeping before and after subsampling --------------------
        for cind in range(3):
            roi_mask = (cur_roi_labels == (cind+1))
            cur_max_overlaps = max_overlaps[roi_mask]
            # before subsampling
            classwise_max_iou_bfs[cind] = torch.cat([cur_max_overlaps, 
                                                            cur_max_overlaps.new_full((cur_roi.shape[0] - cur_max_overlaps.shape[0],), -1)]) # padded with -1 to keep 0   
            # after subsampling
            roi_mask = (cur_roi_labels[sampled_inds] == (cind+1))
            cur_max_overlaps = max_overlaps[sampled_inds][roi_mask]
            classwise_max_iou_afs[cind] = torch.cat([cur_max_overlaps, 
                                                            cur_max_overlaps.new_full((sampled_inds.shape[0] - cur_max_overlaps.shape[0],), -1)]) # padded with -1 to keep 0

        return sampled_inds, reg_valid_mask, cls_labels, roi_ious, gt_assignment, interval_mask, classwise_max_iou_bfs, classwise_max_iou_afs
    
    def classaware_subsampler(self, batch_dict, index):

        iou_fg_thresh = batch_dict['iou_fg_thresh']
        cur_roi = batch_dict['rois'][index]
        cur_gt_boxes = batch_dict['gt_boxes'][index]
        cur_roi_labels = batch_dict['roi_labels'][index]
        classwise_max_iou_bfs = {}
        classwise_max_iou_afs = {}

        k = cur_gt_boxes.__len__() - 1
        while k >= 0 and cur_gt_boxes[k].sum() == 0:
            k -= 1
        cur_gt_boxes = cur_gt_boxes[:k + 1]
        cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(
            cur_gt_boxes) == 0 else cur_gt_boxes
            

        if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=cur_roi, roi_labels=cur_roi_labels,
                gt_boxes=cur_gt_boxes[:, 0:7], gt_labels=cur_gt_boxes[:, -1].long())
                
        else:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt_boxes[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            
        
        # -------------------- class aware subsampling-----------------------------------------------------#
        fg_inds_class_aw, bg_inds_class_aw = [], []
        for cind in range(3):
            roi_mask = (cur_roi_labels == (cind + 1))
            if not roi_mask.sum(): continue
            cur_max_overlaps = max_overlaps[roi_mask]
            fg_inds, bg_inds = self.subsample_rois(max_overlaps= cur_max_overlaps,
                                                    CLS_FG_THRESH=iou_fg_thresh[cind])
            fg_inds_class_aw.append(fg_inds) 
            bg_inds_class_aw.append(bg_inds)
    
        
        
        # FG
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_inds_all = torch.cat(fg_inds_class_aw, dim=0)  # NOTE: class aware is considered only
        
        # There are FGs with repetion, turn on to keep only unique instances (default: Disabled)
        if self.roi_sampler_cfg.get("ENFORCE_UNIQUE_FG_REGIONS", False):
            fg_inds_all = torch.unique(fg_inds_all)

        # discard samples if exceeds limit using max_overlaps
        if len(fg_inds_all) > fg_rois_per_image:
            sorted_fg_inds = fg_inds_all[torch.argsort(max_overlaps[fg_inds_all], descending=True)]
            fg_inds_all = sorted_fg_inds[:fg_rois_per_image]

        # BG
        bg_rois_per_image = self.roi_sampler_cfg.ROI_PER_IMAGE - len(fg_inds_all)

        bg_inds_all = torch.cat(bg_inds_class_aw, dim=0) # NOTE: class aware is considered only
        # discard samples if exceeds limit using max_overlaps
        if len(bg_inds_all) > bg_rois_per_image:
            sorted_bg_inds = bg_inds_all[torch.argsort(max_overlaps[bg_inds_all], descending=True)]
            bg_inds_all = sorted_bg_inds[:bg_rois_per_image]


        sampled_inds = torch.cat((fg_inds_all, bg_inds_all), dim=0)[:self.roi_sampler_cfg.ROI_PER_IMAGE]
        # ------------------------------------------------------------------------------------------------#


        roi_ious = max_overlaps[sampled_inds]
        reg_valid_mask = (roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()         # regression valid mask
        iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH

        
        if self.roi_sampler_cfg.get("ENABLE_CLASSAWRAE_IOU_ASSIGNEMENT", False):
            # ------- class-aware label assignment, could be fixed or adaptive -----------------------------#
            
            cls_labels = torch.zeros_like(roi_ious)
            interval_mask = torch.zeros_like(roi_ious, dtype=bool)
            ignore_mask = torch.eq(roi_ious, 0).all(dim=-1)
            cls_labels[ignore_mask] = -1
            
            for cind in range(3):
                classwise_mask = cur_roi_labels[sampled_inds] == (cind+1)
                classwise_roi_ious = roi_ious[classwise_mask]
                fg_mask = classwise_roi_ious > iou_fg_thresh[cind]
                bg_mask = classwise_roi_ious < iou_bg_thresh
                interval_mask[classwise_mask] = (fg_mask == 0) & (bg_mask == 0)
                cls_labels[classwise_mask] = (fg_mask > 0).float()
                
                if self.roi_sampler_cfg.get("CALIBRATED_IOUS", False):
                    cls_labels[classwise_mask][interval_mask[classwise_mask]] = \
                        (classwise_roi_ious[interval_mask[classwise_mask]] - iou_bg_thresh) / (iou_fg_thresh[cind] - iou_bg_thresh)
        else:
            # ---------------------- default roi_iou based assignement ------------------

            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = roi_ious > iou_fg_thresh
            bg_mask = roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)
            cls_labels = (fg_mask > 0).float()

            cls_labels[interval_mask] = \
                (roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)                
        
                

        # ------------------- Temprorialy added for record keeping before and after subsampling --------------------
        for cind in range(3):
            roi_mask = (cur_roi_labels == (cind+1))
            cur_max_overlaps = max_overlaps[roi_mask]
            # before subsampling
            classwise_max_iou_bfs[cind] = torch.cat([cur_max_overlaps, 
                                                            cur_max_overlaps.new_full((cur_roi.shape[0] - cur_max_overlaps.shape[0],), -1)]) # padded with -1 to keep 0   
            # after subsampling
            roi_mask = (cur_roi_labels[sampled_inds] == (cind+1))
            cur_max_overlaps = max_overlaps[sampled_inds][roi_mask]
            classwise_max_iou_afs[cind] = torch.cat([cur_max_overlaps, 
                                                            cur_max_overlaps.new_full((sampled_inds.shape[0] - cur_max_overlaps.shape[0],), -1)]) # padded with -1 to keep 0

        return sampled_inds, reg_valid_mask, cls_labels, roi_ious, gt_assignment, interval_mask, classwise_max_iou_bfs, classwise_max_iou_afs
    

    
    def subsample_rois(self, max_overlaps, 
                    FG_RATIO=None, 
                    ROI_PER_IMAGE=None, 
                    REG_FG_THRESH=None, 
                    CLS_FG_THRESH=None, 
                    CLS_BG_THRESH_LO=None,
                    HARD_BG_RATIO=None):
        
        if FG_RATIO is None:            FG_RATIO = self.roi_sampler_cfg.FG_RATIO
        if ROI_PER_IMAGE is None:       ROI_PER_IMAGE = self.roi_sampler_cfg.ROI_PER_IMAGE
        if REG_FG_THRESH is None:       REG_FG_THRESH = self.roi_sampler_cfg.REG_FG_THRESH
        if CLS_FG_THRESH is None:       CLS_FG_THRESH = self.roi_sampler_cfg.CLS_FG_THRESH
        if CLS_BG_THRESH_LO is None:    CLS_BG_THRESH_LO = self.roi_sampler_cfg.CLS_BG_THRESH_LO
        if HARD_BG_RATIO is None:       HARD_BG_RATIO = self.roi_sampler_cfg.HARD_BG_RATIO

        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(FG_RATIO * ROI_PER_IMAGE))
        fg_thresh = min(REG_FG_THRESH, CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)  # > 0.55
        easy_bg_inds = ((max_overlaps < CLS_BG_THRESH_LO)).nonzero().view(-1)  # < 0.1
        hard_bg_inds = ((max_overlaps < REG_FG_THRESH) &
                (max_overlaps >= CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        #sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return fg_inds, bg_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, 8)
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment