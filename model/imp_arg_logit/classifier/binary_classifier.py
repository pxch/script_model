import pickle as pkl
from collections import defaultdict
from copy import deepcopy
from os.path import exists

import numpy as np
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

from common.imp_arg import ImplicitArgumentDataset
from common.imp_arg import helper
from model.imp_arg_logit.features import PredicateFeatureSet
from utils import check_type, log


class BinarySample(object):
    def __init__(self, pred_pointer, fold_idx, arg_label, feature_set, label):
        self.pred_pointer = pred_pointer
        self.fold_idx = fold_idx
        self.arg_label = arg_label
        self.feature_set = feature_set
        self.label = label

    @classmethod
    def from_proposition(cls, proposition, corenlp_mapping, fold_idx_mapping,
                         use_list=False):
        sample_list = []
        pred_pointer = proposition.pred_pointer
        idx_mapping, doc = corenlp_mapping[pred_pointer.fileid]
        fold_idx = fold_idx_mapping[str(pred_pointer)]

        predicate_feature_set = PredicateFeatureSet.build(
            proposition, doc, idx_mapping, use_list)

        missing_labels = proposition.missing_labels()

        for arg_label in missing_labels:
            feature_set = deepcopy(predicate_feature_set)
            feature_set.set_imp_arg(arg_label)
            if arg_label in proposition.imp_args:
                label = 1
            else:
                label = 0

            sample_list.append(
                cls(pred_pointer, fold_idx, arg_label, feature_set, label))
        return sample_list


class BinaryClassifier(object):
    def __init__(self, n_splits=10):
        # number of splits in cross validation
        self.n_splits = n_splits

        # list of classification samples
        self.sample_list = []

        # Preprocessed features of all samples, should be a sparse matrix
        self.features = None
        # Labels of all samples, should be a numpy array
        self.labels = None

        # List of predicted labels of each sample
        self.labels_pred = []
        # Mapping from predicate pointer to list of predicted missing labels:
        self.missing_labels_mapping = None

        # Grid of hyper parameters to search
        self.param_grid = None

        # list of trained logistic regression models for each fold
        self.logit_list = []
        # list of best parameters for each fold
        self.best_param_list = []
        # list of best validation f1 scores for each fold
        self.best_val_f1_list = []

    def read_dataset(self, dataset, use_list=False, save_path=None):
        check_type(dataset, ImplicitArgumentDataset)
        if not exists(helper.propositions_path):
            dataset.build_propositions(save_path=helper.propositions_path)
        else:
            dataset.load_propositions(helper.propositions_path)

        corenlp_mapping = dataset.corenlp_mapping

        log.info('Building predicate feature set for every missing argument '
                 'label of every propositions')

        fold_idx_mapping = {}

        dataset.create_train_test_folds(n_splits=self.n_splits)
        for fold_idx in range(self.n_splits):
            test_fold = dataset.get_test_fold(fold_idx)
            for test_idx in test_fold:
                key = str(dataset.propositions[test_idx].pred_pointer)
                fold_idx_mapping[key] = fold_idx

        self.sample_list = []

        for proposition in dataset.propositions:
            sample_list = BinarySample.from_proposition(
                proposition, corenlp_mapping, fold_idx_mapping, use_list)
            self.sample_list.extend(sample_list)

        if save_path:
            log.info('Saving sample list to {}'.format(save_path))
            pkl.dump(self.sample_list, open(save_path, 'w'))

    def load_sample_list(self, sample_list_path):
        log.info('Loading sample list from {}'.format(sample_list_path))
        self.sample_list = pkl.load(open(sample_list_path, 'r'))

    def preprocess_features(self, featurizer='one_hot'):
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
        self.logit_list = []
        for fold_idx in range(self.n_splits):
            logit = LogisticRegression(fit_intercept=fit_intercept)
            if not tune_w:
                logit.set_params(class_weight='balanced')
            self.logit_list.append(logit)

        if tune_w:
            self.param_grid = ParameterGrid({
                'C': [2 ** x for x in range(-4, 1)],
                'class_weight': [{0: 1, 1: 2 ** x} for x in range(0, 10)]
            })
        else:
            self.param_grid = ParameterGrid({
                'C': [10 ** x for x in range(-4, 5)]
            })

    def cross_validation(self, use_val=False, verbose=False):
        self.labels_pred = np.asarray([-1] * len(self.labels))
        self.best_param_list = []
        self.best_val_f1_list = []

        fold_idx_list = [sample.fold_idx for sample in self.sample_list]

        for fold_idx in range(self.n_splits):
            test_indices = \
                [i for i in range(len(fold_idx_list))
                 if fold_idx_list[i] == fold_idx]

            test_features = self.features[test_indices]
            test_gold = self.labels[test_indices]

            if use_val:
                if fold_idx == 0:
                    val_fold = 9
                else:
                    val_fold = fold_idx - 1
                log.info(
                    'Test fold #{}, use fold #{} as validation'.format(
                        fold_idx, val_fold))

                val_indices = \
                    [i for i in range(len(fold_idx_list))
                     if fold_idx_list[i] == val_fold]
                train_indices = \
                    [i for i in range(len(fold_idx_list))
                     if fold_idx_list[i] != fold_idx
                     and fold_idx_list[i] != val_fold]

                val_features = self.features[val_indices]
                val_gold = self.labels[val_indices]

            else:
                log.info(
                    'Test fold #{}, use all training as validation'.format(
                        fold_idx))

                train_indices = \
                    [i for i in range(len(fold_idx_list))
                     if fold_idx_list[i] != fold_idx]

                val_features = self.features[train_indices]
                val_gold = self.labels[train_indices]

            train_features = self.features[train_indices]
            train_gold = self.labels[train_indices]

            best_param = {}
            best_val_f1 = 0

            logit = self.logit_list[fold_idx]

            for param in self.param_grid:
                logit.set_params(**param)

                logit.fit(train_features, train_gold)

                val_pred = logit.predict(val_features)

                val_f1 = self.eval(
                    val_pred,
                    val_gold,
                    'Fold #{}, params = {}'.format(fold_idx, param),
                    verbose=verbose
                )

                if val_f1 > best_val_f1:
                    best_param = param
                    best_val_f1 = val_f1

            log.info(
                'Selecting best param = {}, with validation f1 = {}'.format(
                    best_param, best_val_f1))
            self.best_param_list.append(best_param)
            self.best_val_f1_list.append(best_val_f1)

            logit.set_params(**best_param)
            logit.fit(train_features, train_gold)

            test_pred = logit.predict(test_features)

            self.eval(
                test_pred,
                test_gold,
                'Test fold #{}'.format(fold_idx),
                verbose=True
            )

            self.labels_pred[test_indices] = test_pred

        self.eval(self.labels_pred, self.labels, 'Total', verbose=True)

    @staticmethod
    def eval(predict, gold, log_msg, verbose=True):
        tp = sum([1 for y1, y2 in zip(predict, gold) if y1 == 1 and y2 == 1])
        fp = sum([1 for y1, y2 in zip(predict, gold) if y1 == 1 and y2 == 0])
        fn = sum([1 for y1, y2 in zip(predict, gold) if y1 == 0 and y2 == 1])
        tn = sum([1 for y1, y2 in zip(predict, gold) if y1 == 0 and y2 == 0])

        accuracy = 100. * (tp + tn) / len(gold)
        precision = 100. * tp / (tp + fp) if tp + fp > 0 else 0
        recall = 100. * tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2. * precision * recall / (precision + recall) \
            if precision + recall > 0 else 0

        if verbose:
            put_log = log.info
        else:
            put_log = log.debug

        put_log(
            '{}: precision = {:6.2f}, recall = {:6.2f}, f1 = {:6.2f}, '
            'accuracy = {} / {} = {:6.2f} %'.format(
                log_msg, precision, recall, f1, tp + tn, len(gold), accuracy))

        return f1

    def predict_missing_labels(self):
        self.missing_labels_mapping = defaultdict(list)

        for sample, label_pred in zip(self.sample_list, self.labels_pred):
            key = str(sample.pred_pointer)
            if label_pred:
                self.missing_labels_mapping[key].append(sample.arg_label)

    def save_missing_labels(self, save_path):
        log.info('Saving predicted missing labels to {}'.format(save_path))
        pkl.dump(self.missing_labels_mapping, open(save_path, 'w'))

    def save_models(self, save_path):
        log.info('Saving models to {}'.format(save_path))
        models = {
            'labels_pred': self.labels_pred,
            'logit_list': self.logit_list,
            'best_param_list': self.best_param_list,
            'best_val_f1_list': self.best_val_f1_list}
        pkl.dump(models, open(save_path, 'w'))
