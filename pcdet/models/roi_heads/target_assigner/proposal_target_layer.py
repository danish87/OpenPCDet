import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg
        self.sample_num_rois = self.roi_sampler_cfg.NUM_ROI_OVERLAP_STUDENT_TEACHER

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """

        
        (batch_rois, 
            batch_gt_of_rois, 
                batch_roi_ious, batch_roi_scores, batch_roi_labels) = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        if "rois_teacher" in batch_dict:
            # TODO:OPTION1:
            # find overlap b/w student and teacher's proposals
            # find overlap b/w matched and GT
            # Apply sampling

            # OPTION2:
            # find the overlaped proosals b/w student vs GT and perform sampling
            # find the overlaped proosals b/w teacher vs GT anb perform sampling
            # find overlap b/w match1 and match2
            keep_rois = batch_dict['rois'].data.clone()
            keep_roi_scores = batch_dict['roi_scores'].data.clone()
            keep_roi_labels = batch_dict['roi_labels'].data.clone()

            batch_dict['rois'] = batch_dict['rois_teacher']
            batch_dict['roi_scores'] = batch_dict['roi_scores_teacher']
            batch_dict['roi_labels'] = batch_dict['roi_labels_teacher']

            (batch_rois_teacher, 
                batch_gt_of_rois_teacher, 
                batch_roi_ious_teacher, batch_roi_scores_teacher, batch_roi_labels_teacher) = self.sample_rois_for_rcnn(
                batch_dict=batch_dict
            )
            batch_dict['rois'] = keep_rois
            batch_dict['roi_scores'] = keep_roi_scores
            batch_dict['roi_labels'] = keep_roi_labels

            code_size = batch_dict['rois'].shape[-1]
            sampled_batch_rois = batch_dict['rois'].new_zeros(batch_dict['batch_size'], self.sample_num_rois, code_size)
            sampled_batch_gt_of_rois = batch_dict['rois'].new_zeros(batch_dict['batch_size'], self.sample_num_rois, code_size + 1)
            sampled_batch_roi_ious = batch_dict['rois'].new_zeros(batch_dict['batch_size'], self.sample_num_rois)
            sampled_batch_roi_scores = batch_dict['rois'].new_zeros(batch_dict['batch_size'], self.sample_num_rois)
            sampled_batch_roi_labels = batch_dict['rois'].new_zeros((batch_dict['batch_size'], self.sample_num_rois), dtype=torch.long)

            sampled_batch_rois_teacher = batch_dict['rois_teacher'].new_zeros(batch_dict['batch_size'], self.sample_num_rois, code_size)
            sampled_batch_gt_of_rois_teacher = batch_dict['rois_teacher'].new_zeros(batch_dict['batch_size'], self.sample_num_rois, code_size + 1)
            sampled_batch_roi_ious_teacher = batch_dict['rois_teacher'].new_zeros(batch_dict['batch_size'], self.sample_num_rois)
            sampled_batch_roi_scores_teacher = batch_dict['rois_teacher'].new_zeros(batch_dict['batch_size'], self.sample_num_rois)
            sampled_batch_roi_labels_teacher = batch_dict['rois_teacher'].new_zeros((batch_dict['batch_size'], self.sample_num_rois), dtype=torch.long)

            # Find overlap b/w student and teacher's proposals
            for index in range(batch_dict['batch_size']):
                (cur_roi, cur_roi_teacher,
                cur_gt, cur_gt_teacher, 
                cur_roi_labels, cur_roi_labels_teacher, 
                cur_roi_scores, cur_roi_scores_teacher) = \
                (batch_rois[index],  batch_rois_teacher[index],
                batch_gt_of_rois[index], batch_gt_of_rois_teacher[index],
                batch_roi_labels[index], batch_roi_labels_teacher[index],
                batch_roi_scores[index], batch_roi_scores_teacher[index])

                k = cur_roi_teacher.__len__() - 1
                while k >= 0 and cur_roi_teacher[k].sum() == 0:
                    k -= 1
                cur_roi_teacher = cur_roi_teacher[:k + 1]
                cur_roi_teacher = cur_roi_teacher.new_zeros((1, cur_roi_teacher.shape[1])) if len(cur_roi_teacher) == 0 else cur_roi_teacher

                
                if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                    max_overlaps, t_assignment = self.get_max_iou_with_same_class(
                        rois=cur_roi, roi_labels=cur_roi_labels,
                        gt_boxes=cur_roi_teacher, gt_labels=cur_roi_labels_teacher
                    )
                else:
                    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_roi_teacher)  # (M, N)
                    max_overlaps, t_assignment = torch.max(iou3d, dim=1)
                
                # Sort max_overlaps and sample the top 64 (NUM_ROI_OVERLAP_STUDENT_TEACHER) rois.
                sorted_max_overlap, inds_max_overlap = torch.sort(max_overlaps)
                sampled_inds = inds_max_overlap[:self.sample_num_rois] 
                
                # We can also try to sample the rois based on an overlap threshold instead of selecting the top 64. But, dim mismatch occurs.
                # sampled_inds = (max_overlaps > self.roi_sampler_cfg.STUDENT_TEACHER_ROI_OVERLAP_THRESH).nonzero()[:, 0]
                
                sampled_batch_rois[index] = cur_roi[sampled_inds]
                sampled_batch_rois_teacher[index] = cur_roi_teacher[t_assignment[sampled_inds]]

                sampled_batch_gt_of_rois[index] = cur_gt[sampled_inds]
                sampled_batch_gt_of_rois_teacher[index] = cur_gt_teacher[t_assignment[sampled_inds]]
                
                sampled_batch_roi_labels[index] = cur_roi_labels[sampled_inds]
                sampled_batch_roi_labels_teacher[index] = cur_roi_labels_teacher[t_assignment[sampled_inds]]
                
                sampled_batch_roi_ious[index] = max_overlaps[sampled_inds]
                sampled_batch_roi_ious_teacher[index] = max_overlaps[t_assignment[sampled_inds]]
                
                sampled_batch_roi_scores[index] = cur_roi_scores[sampled_inds]
                sampled_batch_roi_scores_teacher[index] = cur_roi_scores_teacher[t_assignment[sampled_inds]]
                
        # unsupervised R-CNN classification loss
        # regression valid mask
        reg_valid_mask = (sampled_batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()
        #TODO reg_valid_mask for teacher proposals
        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':

            batch_cls_labels = (sampled_batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (sampled_batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (sampled_batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1  # all preds inside middle-region (0.25-0.75) are considered ignored.

        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou': #or 'soft_teacher':
            
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH 
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = sampled_batch_roi_ious > iou_fg_thresh # >0.75
            bg_mask = sampled_batch_roi_ious < iou_bg_thresh # <0.25
            interval_mask = (fg_mask == 0) & (bg_mask == 0) 
            batch_cls_labels = (fg_mask > 0).float() # 1 or zero
            

            #! @Farzad: rcnn cls labels are created here. The inverval mask or hard bgs are weighted (linearly) according to
            # their iou-score with pseudo-labels (becasue of sampled_batch_roi_ious in nominator)
            
            if "rois_teacher" in batch_dict:
                

                #! @Farzad: it can be combination of student roi_iou  and teacher roi_iou
                if self.roi_sampler_cfg.ENABLE_HYBRID:
                    
                    roi_iou_score = \
                        (sampled_batch_roi_ious - iou_bg_thresh) / (iou_fg_thresh-iou_bg_thresh)
                    roi_iou_score_teacher = \
                        (sampled_batch_roi_ious_teacher - iou_bg_thresh) / (iou_fg_thresh-iou_bg_thresh)
                    
                    batch_cls_labels[interval_mask]= \
                        (
                            (self.roi_sampler_cfg.CLS_WEIGHT*roi_iou_score_teacher) + 
                            ((1-self.roi_sampler_cfg.CLS_WEIGHT)*roi_iou_score)

                        )[interval_mask]
                    
                else:
                    batch_cls_labels[interval_mask] = sampled_batch_roi_ious_teacher[interval_mask]


            else:
                
                batch_cls_labels[interval_mask] = \
                (sampled_batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'raw_roi_iou': # st3d settings
            batch_cls_labels = sampled_batch_roi_ious
        
        else:
            raise NotImplementedError

        targets_dict = {'rois': sampled_batch_rois, 'gt_of_rois': sampled_batch_gt_of_rois, 'gt_iou_of_rois': sampled_batch_roi_ious,
                        'roi_scores': sampled_batch_roi_scores, 'roi_labels': sampled_batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels,
                        'rois_teacher': sampled_batch_rois_teacher, 'gt_of_rois_teacher': sampled_batch_gt_of_rois_teacher, 
                        'gt_iou_of_rois_teacher': sampled_batch_roi_ious_teacher,
                        'roi_scores_teacher': sampled_batch_roi_scores_teacher, 'roi_labels_teacher': sampled_batch_roi_labels_teacher,
                        }

        return targets_dict

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
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)


            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)  # > 0.55
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)  # < 0.1
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

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
            gt_boxes: (N, )
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
