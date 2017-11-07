from collections import OrderedDict


class FeatureSet(object):
    def __init__(self, feature_map):
        self._feature_list = []
        self._feature_map = OrderedDict()
        for feature, value in feature_map.items():
            self._feature_list.append(feature)
            self._feature_map[feature] = value

    @property
    def feature_list(self):
        return self._feature_list

    @property
    def feature_map(self):
        return self._feature_map

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

    def get_subset(self, feature_sublist=None):
        feature_map = OrderedDict()
        for feature in feature_sublist:
            assert feature in self.feature_list
            feature_map[feature] = self.get_feature(feature)
        return FeatureSet(feature_map)

    @classmethod
    def merge(cls, *args):
        feature_map = OrderedDict()

        for feature_set in args:
            for feature in feature_set.feature_list:
                feature_map[feature] = feature_set.get_feature(feature)

        return cls(feature_map)
