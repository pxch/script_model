import abc
from collections import OrderedDict


class BaseFeatureSet(object):
    __metaclass__ = abc.ABCMeta

    feature_list = None

    def __init__(self, **kwargs):
        self._feature_map = OrderedDict()
        self._feature_val_list = []
        for feature in self.feature_list:
            assert feature in kwargs, 'feature {} not found'.format(feature)
            self._feature_map[feature] = kwargs[feature]
            self._feature_val_list.append(kwargs[feature])

    @property
    def feature_map(self):
        return self._feature_map

    @property
    def feature_val_list(self):
        return self._feature_val_list

    def get_feature(self, feature):
        assert feature in self.feature_list, \
            'features not found: {}'.format(feature)
        return self.feature_map[feature]

    def pretty_print(self):
        return '\n'.join(
            '{}\t{}'.format(feature, value) for feature, value
            in self.feature_map.items())

    def iteritems(self):
        key_val_list = []
        for key, val in self._feature_map.items():
            if isinstance(val, list):
                for item in val:
                    if (key, item) not in key_val_list:
                        key_val_list.append((key, item))
            else:
                if (key, val) not in key_val_list:
                    key_val_list.append((key, val))
        for key, val in key_val_list:
            yield key, val
