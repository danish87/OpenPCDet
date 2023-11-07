from .adaptive_thresholding import AdaptiveThresholding
__all__ = {
    'AdaptiveThresholding': AdaptiveThresholding
    }

class ThresholdingAlgRegistry(object):
    def __init__(self, **kwargs):
        self._algorithms = {}
    def register(self, name, **kwargs):
        if name in self._algorithms.keys():
            thresh_alg = self._algorithms[name]
        else:
            thresh_alg = __all__[name](**kwargs)
            self._algorithms[name] = thresh_alg
        return thresh_alg
    def get(self, name):
        if name in self._algorithms.keys():
            return self._algorithms[name]
    def __getitem__(self, key):
        return self._algorithms[key]

    def __contains__(self, key):
        return key in self._algorithms

    def keys(self):
        return self._algorithms.keys()

thresh_registry = ThresholdingAlgRegistry()