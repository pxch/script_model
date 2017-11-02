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

        # preprocessed features of all samples, a sparse matrix
        self.features = None
        # labels of all samples, a numpy array
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

        raw_features = [sample.feature_set for sample in self.sample_list]
        if featurizer == 'one_hot':
            vec = DictVectorizer()
            self.features = vec.fit_transform(raw_features)
        elif featurizer == 'hash':
            hasher = FeatureHasher()
            self.features = hasher.transform(raw_features)
        else:
            raise ValueError('Unrecognized featurizer: ' + featurizer)

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
            # grid['C'] = [1]
            grid['class_weight'] = ['balanced']
        self.param_grid = ParameterGrid(grid)

    @abc.abstractmethod
    def reset_states(self):
        pass

    def cross_validation(self, use_val=False, verbose=False):
        self.reset_states()

        for test_fold_idx in range(self.n_splits):
            log.info('=' * 20)
            if use_val:
                if test_fold_idx == 0:
                    val_fold_idx = 9
                else:
                    val_fold_idx = test_fold_idx - 1

                val_fold_indices = [val_fold_idx]

                train_sample_indices = \
                    [sample_idx for sample_idx, fold_idx
                     in enumerate(self.sample_idx_to_fold_idx)
                     if fold_idx != test_fold_idx
                     and fold_idx != val_fold_idx]

            else:
                val_fold_indices = \
                    [fi for fi in range(self.n_splits) if fi != test_fold_idx]

                train_sample_indices = \
                    [sample_idx for sample_idx, fold_idx
                     in enumerate(self.sample_idx_to_fold_idx)
                     if fold_idx != test_fold_idx]

            log.info(
                'Test fold #{}, use fold #{} as validation'.format(
                    test_fold_idx, val_fold_indices))

            train_features = self.features[train_sample_indices]
            train_gold = self.labels[train_sample_indices]

            self.train_model(train_features, train_gold, val_fold_indices,
                             test_fold_idx, verbose=verbose)

    @abc.abstractmethod
    def train_model(self, train_features, train_gold, val_fold_indices,
                    test_fold_idx, verbose=False):
        pass

    def save_models(self, save_path):
        log.info('Saving models to {}'.format(save_path))
        pkl.dump(self.model_list, open(save_path, 'w'))

    @abc.abstractmethod
    def print_stats(self, fout=None):
        pass
