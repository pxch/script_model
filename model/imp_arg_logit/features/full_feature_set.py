from collections import OrderedDict

from .base_feature_set import BaseFeatureSet
from .filler_feature_set import FillerFeatureSet
from .predicate_feature_set import PredicateFeatureSet


class FullFeatureSet(BaseFeatureSet):
    feature_list = \
        PredicateFeatureSet.feature_list + FillerFeatureSet.feature_list

    @classmethod
    def merge(cls, predicate_feature_set, filler_feature_set):
        kwargs = OrderedDict()

        for key in predicate_feature_set.feature_list:
            kwargs[key] = predicate_feature_set.get_feature(key)
        for key in filler_feature_set.feature_list:
            kwargs[key] = filler_feature_set.get_feature(key)

        return cls(**kwargs)
