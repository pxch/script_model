import abc
import pickle as pkl
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import ParameterGrid

from utils import log


class BaseClassifier(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_splits=10):
        # number of folds in cross validation
        self.n_splits = n_splits

        # implicit argument dataset
        self.dataset = None

        # mapping from pred_pointers to nominal predicates
        self.pred_pointer_to_n_pred = None

        # mappings between pred_pointers and fold indices
        self.pred_pointer_to_fold_idx = None
        self.fold_idx_to_pred_pointer = None

        # list of classification samples
        self.sample_list = []

        # mapping from sample indices to fold indices
        self.sample_idx_to_fold_idx = None

        # list of raw features of all samples,
        # can be transformed to vectorized features by self.transformer
        self.raw_features = None
        # transformer for raw features,
        # can be either a DictVectorizer or aFeatureHasher
        self.transformer = None

        # list of preprocessed features of all samples, a sparse matrix
        self.features = None
        # list of labels of all samples, a numpy array
        self.labels = None

        # grid of hyper parameters to search
        self.param_grid = None

        # list of model states for each fold, including
        # content of each state depends on classifier
        self.model_list = []

    @abc.abstractmethod
    def read_dataset(self, dataset):
        pass

    def index_dataset(self):
        log.info('Building mapping from pred_pointer to n_pred')
        self.pred_pointer_to_n_pred = {}
        for proposition in self.dataset.propositions:
            pred_pointer = str(proposition.pred_pointer)
            n_pred = proposition.n_pred
            self.pred_pointer_to_n_pred[pred_pointer] = n_pred

        self.dataset.create_train_test_folds(n_splits=self.n_splits)

        log.info('Building mapping from pred_pointer to fold_idx')
        self.pred_pointer_to_fold_idx = {}
        for fold_idx in range(self.n_splits):
            test_indices = self.dataset.get_test_fold(fold_idx)
            for proposition_idx in test_indices:
                pred_pointer = str(
                    self.dataset.propositions[proposition_idx].pred_pointer)
                self.pred_pointer_to_fold_idx[pred_pointer] = fold_idx

        log.info('Building mapping from fold_idx to list of pred_pointer')
        self.fold_idx_to_pred_pointer = defaultdict(list)
        for proposition in self.dataset.propositions:
            pred_pointer = str(proposition.pred_pointer)
            fold_idx = self.pred_pointer_to_fold_idx[pred_pointer]
            self.fold_idx_to_pred_pointer[fold_idx].append(pred_pointer)

    @abc.abstractmethod
    def build_sample_list(self):
        pass

    def load_sample_list(self, sample_list_path):
        log.info('Loading sample list from {}'.format(sample_list_path))
        self.sample_list = pkl.load(open(sample_list_path, 'r'))
        log.info('Done...')

    def index_sample_list(self):
        self.sample_idx_to_fold_idx = []
        log.info('Building mapping from sample idx to fold_idx')
        self.sample_idx_to_fold_idx = \
            [sample.fold_idx for sample in self.sample_list]

    def preprocess_features(self, featurizer='one_hot'):
        log.info('Processing features with {} featurizer'.format(featurizer))

        self.raw_features = [sample.feature_set for sample in self.sample_list]
        if featurizer == 'one_hot':
            self.transformer = DictVectorizer()
            self.transformer.fit(self.raw_features)
        elif featurizer == 'hash':
            self.transformer = FeatureHasher()
        else:
            raise ValueError('Unrecognized featurizer: ' + featurizer)

        self.features = self.transformer.transform(self.raw_features)

        labels = [sample.label for sample in self.sample_list]
        self.labels = np.asarray(labels)

    def set_hyper_parameter(self, fit_intercept=True, tune_w=False):
        log.info('Setting hyperparameters range for tuning')
        grid = {'fit_intercept': [fit_intercept]}
        if tune_w:
            grid['C'] = [2 ** x for x in range(-4, 1)]
            grid['class_weight'] = [{0: 1, 1: 2 ** x} for x in range(0, 10)]
        else:
            grid['C'] = [10 ** x for x in range(-4, 5)]
            grid['class_weight'] = ['balanced']
        self.param_grid = ParameterGrid(grid)

    @abc.abstractmethod
    def set_states(self, states):
        pass

    def save_models(self, save_path):
        log.info('Saving models to {}'.format(save_path))
        pkl.dump(self.model_list, open(save_path, 'w'))

    @abc.abstractmethod
    def print_stats(self, fout=None):
        pass
