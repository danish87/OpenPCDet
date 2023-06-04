from .detector3d_template import Detector3DTemplate
from ...utils.stats_utils import KITTIEvalMetrics, PredQualityMetrics
from torchmetrics.collections import MetricCollection
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

    def forward(self, batch_dict):
        batch_dict['metric_registry'] = self.metric_registry
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
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
            rcnn_entropy =self.roi_head.calc_entropy(
                lambda_=self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("RCNN_ENTROPY_REG_LAMBDA", 0.25)
                ).mean()
            tb_dict['rcnn_entropy'] = rcnn_entropy
            loss += rcnn_entropy

        # RPN Entropy Regularization
        if self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("ENABLE_RPN_ENTROPY_REG", False):
            rpn_entropy =self.dense_head.calc_entropy(
                lambda_=self.model_cfg.ROI_HEAD.LOSS_CONFIG.get("RPN_ENTROPY_REG_LAMBDA", 0.005)
                ).mean()
            tb_dict['rpn_entropy'] = rpn_entropy
            loss += rpn_entropy
        
        return loss, tb_dict, disp_dict

    def compute_metrics(self, registry, tag):
        results = registry.get(tag).compute()
        tag = tag + "/" if tag else ''
        metrics = {tag + key: val for key, val in results.items()}
        return metrics