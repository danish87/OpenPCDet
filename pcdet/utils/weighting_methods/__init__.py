from .freematch import FreeMatchThreshold
from .adamatch import AdaMatchThreshold
from .consistant_teacher import AdaptiveThresholdGMM
from .softmatch import SoftMatchThreshold

__all__ = {
    'FreeMatchThreshold': FreeMatchThreshold,
    'AdaMatchThreshold': AdaMatchThreshold,
    'AdaptiveThresholdGMM': AdaptiveThresholdGMM,
    # 'SoftMatchThreshold': SoftMatchThreshold
    }


def build_thresholding_method(tag, dataset, config):
    model = __all__[config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.NAME](
        tag=tag, dataset=dataset, config=config
    )

    return model