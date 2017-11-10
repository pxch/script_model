import pickle as pkl
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter

import numpy as np
from sklearn.linear_model import LogisticRegression
from texttable import Texttable

from common.imp_arg import DiceEvalMetric
from common.imp_arg import ImplicitArgumentDataset
from common.imp_arg import helper
from model.imp_arg_logit.features import FeatureSet
from model.imp_arg_logit.features import FillerFeatureSet, PredicateFeatureSet
from utils import check_type, log, get_file_logger
from .base_classifier import BaseClassifier, BaseModelState


class FullSample(object):
    def __init__(self, pred_pointer, fold_idx, arg_label, candidate_idx,
                 feature_set, label):
        self.pred_pointer = pred_pointer
        self.fold_idx = fold_idx
        self.arg_label = arg_label
        self.candidate_idx = candidate_idx
        self.feature_set = feature_set
        self.label = label

    @classmethod
    def from_proposition(cls, proposition, corenlp_mapping, fold_idx,
                         use_list=False):
        sample_list = []
        pred_pointer = proposition.pred_pointer
        idx_mapping, doc = corenlp_mapping[pred_pointer.fileid]

        base_predicate_feature_set = PredicateFeatureSet.build(
            proposition, doc, idx_mapping, use_list)

        missing_labels = proposition.missing_labels()
        core_arg_mapping = helper.predicate_core_arg_mapping[proposition.v_pred]

        for arg_label in missing_labels:
            predicate_feature_set = deepcopy(base_predicate_feature_set)

            iarg_type = core_arg_mapping[arg_label]
            predicate_feature_set.set_imp_arg(iarg_type)

            for candidate_idx, candidate in enumerate(proposition.candidates):
                filler_feature_set = FillerFeatureSet.build(
                    proposition, doc, arg_label, candidate.arg_pointer,
                    use_list=use_list)

                full_feature_set = FeatureSet.merge(
                    predicate_feature_set, filler_feature_set)

                label = 0
                if arg_label in proposition.imp_args and \
                        candidate.is_oracle(proposition.imp_args[arg_label]):
                    label = 1

                sample_list.append(
                    cls(str(pred_pointer), fold_idx, arg_label, candidate_idx,
                        full_feature_set, label))
        return sample_list


class FullModelState(BaseModelState):
    def __init__(self, logit, param, feature_list, thres, test_fold_idx,
                 val_fold_indices, val_metric):
        super(FullModelState, self).__init__(
            logit=logit, param=param, feature_list=feature_list,
            test_fold_idx=test_fold_idx, val_fold_indices=val_fold_indices,
            val_metric=val_metric)
        self.thres = thres


class FullClassifier(BaseClassifier):
    def __init__(self, n_splits=10):
        super(FullClassifier, self).__init__(n_splits=n_splits)

        # mapping from pred_pointer to dice_score_dict for evaluation purpose
        self.dice_score_dict_mapping = None

        # mapping from pred_pointer to missing_labels,
        # used only when combining with results from BinaryClassifier
        self.missing_labels_mapping = None

        # mapping from pred_pointers to dictionary of sample indices
        self.pred_pointer_to_sample_idx_dict = None
        # mapping from pred_pointers to dictionary of features
        self.pred_pointer_to_features_dict = None

    def read_dataset(self, dataset):
        check_type(dataset, ImplicitArgumentDataset)
        log.info('Reading implicit argument dataset')

        dataset.load_propositions(helper.propositions_w_corenlp_path)
        dataset.load_candidate_dict(helper.candidate_dict_w_corenlp_path)

        dataset.add_candidates()

        self.dataset = dataset

        self.index_dataset()
        self.get_dice_score_dict_mapping()

    def get_dice_score_dict_mapping(self):
        log.info('Building mapping from pred_pointer to dice_score_dict')
        self.dice_score_dict_mapping = {}

        for proposition in self.dataset.propositions:
            pred_pointer = str(proposition.pred_pointer)

            dice_score_dict = proposition.get_dice_score_dict()
            num_gold = proposition.num_imp_args()

            self.dice_score_dict_mapping[pred_pointer] = \
                (num_gold, dice_score_dict)

    def load_missing_labels_mapping(self, missing_labels_mapping_path=None):
        self.missing_labels_mapping = \
            pkl.load(open(missing_labels_mapping_path, 'r'))

    def get_missing_labels(self, pred_pointer):
        missing_labels = None
        if self.missing_labels_mapping:
            missing_labels = self.missing_labels_mapping[pred_pointer]
        return missing_labels

    def build_sample_list(self, use_list=False, save_path=None):
        assert self.dataset is not None
        log.info('Building sample list from every missing argument label '
                 'of every proposition')

        corenlp_mapping = self.dataset.corenlp_mapping

        self.sample_list = []

        for proposition in self.dataset.propositions:
            fold_idx = \
                self.pred_pointer_to_fold_idx[str(proposition.pred_pointer)]
            sample_list = FullSample.from_proposition(
                proposition, corenlp_mapping, fold_idx, use_list)
            self.sample_list.extend(sample_list)

        if save_path:
            log.info('Saving sample list to {}'.format(save_path))
            pkl.dump(self.sample_list, open(save_path, 'w'))

    def index_sample_list(self):
        super(FullClassifier, self).index_sample_list()

        self.pred_pointer_to_sample_idx_dict = {}
        log.info('Building mapping from pred_pointer to dictionary of '
                 'arg_label : sample_idx_list')

        for sample_idx, sample in enumerate(self.sample_list):
            pred_pointer = sample.pred_pointer

            if pred_pointer not in self.pred_pointer_to_sample_idx_dict:
                self.pred_pointer_to_sample_idx_dict[pred_pointer] = \
                    defaultdict(list)

            arg_label = sample.arg_label
            self.pred_pointer_to_sample_idx_dict[pred_pointer][
                arg_label].append(sample_idx)

        for pred_pointer, sample_idx_dict in \
                self.pred_pointer_to_sample_idx_dict.items():
            assert all(
                len(idx_list) == len(sample_idx_dict.values()[0])
                for idx_list in sample_idx_dict.values()[1:])
            flat_list = [idx for idx_list in sample_idx_dict.values()
                         for idx in idx_list]
            assert flat_list == range(
                sample_idx_dict.values()[0][0],
                sample_idx_dict.values()[-1][-1] + 1)

    def index_features(self):
        log.info('Building mapping from pred_pointer to dictionary of '
                 'arg_label : list of features (in a sparse matrix format)')

        self.pred_pointer_to_features_dict = {}

        for pred_pointer, sample_idx_dict in \
                self.pred_pointer_to_sample_idx_dict.items():
            features_dict = {}
            for arg_label, sample_indices in sample_idx_dict.items():
                features_dict[arg_label] = self.features[sample_indices, :]
            self.pred_pointer_to_features_dict[pred_pointer] = \
                features_dict

    def eval_feature_subset(self, logit, feature_list, train_features,
                            train_gold, val_fold_indices):
        train_features_subset = self.get_feature_subset(
            train_features, feature_list)
        logit.fit(train_features_subset, train_gold)

        val_score_matrix_mapping = {}
        for val_fold_idx in val_fold_indices:
            val_score_matrix_mapping.update(
                self.predict_fold(logit, val_fold_idx, post_process=True))

        thres, val_metric = self.search_threshold(val_score_matrix_mapping)
        return thres, val_metric

    def feature_selection(self, logger, logit, train_features, train_gold,
                          val_fold_indices):
        logger.info('=' * 20)
        logger.info('Feature selection under params: {}'.format(
            logit.get_params()))

        full_feature_list = deepcopy(self.feature_list)
        best_feature_list = []
        best_score = -1
        prev_feature_list = []

        while len(full_feature_list) > 0:
            b_feature = None
            b_score = -1
            b_thres = -1
            for feature in full_feature_list:
                feature_list = prev_feature_list + [feature]
                f_thres, f_metric = self.eval_feature_subset(
                    logit, feature_list, train_features, train_gold,
                    val_fold_indices)
                f_score = f_metric.f1()
                logger.debug(
                    'Try adding feature {}, thres = {:.2f}, '
                    'score = {:.2f}'.format(feature, f_thres, f_score))
                if f_score > b_score:
                    b_feature = feature
                    b_score = f_score
                    b_thres = f_thres
            prev_feature_list.append(b_feature)
            prev_score = b_score
            full_feature_list.remove(b_feature)
            logger.info(
                'Adding feature {}, with thres = {:.2f}, '
                'current score = {:.2f}, current set = {}'.format(
                    b_feature, b_thres, prev_score, prev_feature_list))

            if prev_score > best_score:
                while len(prev_feature_list) > 1:
                    r_feature = None
                    r_score = -1
                    r_thres = -1
                    for feature in prev_feature_list:
                        feature_list = deepcopy(prev_feature_list)
                        feature_list.remove(feature)
                        f_thres, f_metric = self.eval_feature_subset(
                            logit, feature_list, train_features,
                            train_gold, val_fold_indices)
                        f_score = f_metric.f1()
                        logger.debug(
                            'Try removing feature {}, thres = {:.2f}, '
                            'score = {:.2f}'.format(feature, f_thres, f_score))
                        if f_score > r_score:
                            r_feature = feature
                            r_score = f_score
                            r_thres = f_thres
                    if r_score > prev_score:
                        prev_feature_list.remove(r_feature)
                        prev_score = r_score
                        full_feature_list.append(r_feature)
                        logger.info(
                            'Removing feature {}, with thres = {:.2f}, '
                            'current score = {:.2f}, current set = {}'.format(
                                r_feature, r_thres, prev_score,
                                prev_feature_list))
                    else:
                        break
                best_feature_list = deepcopy(prev_feature_list)
                best_score = prev_score
                logger.info(
                    'Setting best score = {:.2f}, best feature set {}'.format(
                        best_score, best_feature_list))

        best_thres, best_metric = self.eval_feature_subset(
            logit, best_feature_list, train_features, train_gold,
            val_fold_indices)

        logger.info(
            'Best score = {:.2f}, with thres = {:.2f}, '
            'and feature set {}'.format(
                best_score, best_thres, best_feature_list))

        return best_feature_list, best_thres, best_metric

    def train_fold(self, test_fold_idx, use_val=False, verbose=False,
                   log_to_file=False, log_file_path=None):
        if log_to_file:
            assert log_file_path
            fold_logger = get_file_logger(
                name='fold_{}'.format(test_fold_idx),
                file_path=log_file_path,
                log_level='debug' if verbose else 'info',
                propagate=False)
        else:
            fold_logger = log

        train_fold_indices, val_fold_indices = \
            self.get_train_val_folds(test_fold_idx, use_val=use_val)

        log.info('=' * 20)
        log.info(
            'Test fold #{}, use fold #{} as validation'.format(
                test_fold_idx, val_fold_indices))

        train_sample_indices = self.get_sample_indices_from_folds(
            train_fold_indices)
        train_features = self.features[train_sample_indices, :]
        train_gold = self.labels[train_sample_indices]

        best_param = None
        best_thres = -1
        best_val_f1 = 0
        best_val_metric = None
        best_feature_list = None

        logit = LogisticRegression()

        for param in self.param_grid:
            logit.set_params(**param)

            feature_list, thres, val_metric = self.feature_selection(
                fold_logger, logit, train_features, train_gold,
                val_fold_indices)

            debug_msg = \
                'Validation fold #{}, params = {}, thres = {:.2f}, ' \
                'feature subset = {}, {}'.format(
                    val_fold_indices, param, thres, feature_list, val_metric)

            if verbose:
                log.info(debug_msg)
            else:
                log.debug(debug_msg)

            if val_metric.f1() > best_val_f1:
                best_param = param
                best_thres = thres
                best_val_f1 = val_metric.f1()
                best_val_metric = val_metric
                best_feature_list = feature_list

        log.info('-' * 20)
        log.info(
            'Validation ford #{}, selecting best param = {}, thres = {:.2f} '
            'best feature subset = {}, with validation f1 = {:.2f}'.format(
                val_fold_indices, best_param, best_thres, best_feature_list,
                best_val_f1))

        logit.set_params(**best_param)

        train_features_subset = self.get_feature_subset(
            train_features, best_feature_list)

        logit.fit(train_features_subset, train_gold)

        model_state = FullModelState(
            logit, best_param, best_thres, best_feature_list,
            test_fold_idx, val_fold_indices, best_val_metric)

        return model_state

    def test_all(self):
        self.all_metric_mapping = {}
        assert len(self.model_state_list) == self.n_splits

        for test_fold_idx in range(self.n_splits):
            logit = self.model_state_list[test_fold_idx].logit
            thres = self.model_state_list[test_fold_idx].thres

            test_score_matrix_mapping = self.predict_fold(
                logit, test_fold_idx, post_process=True)

            for pred_pointer, score_matrix in test_score_matrix_mapping.items():
                eval_metric = self.eval_pred_pointer(
                    pred_pointer, score_matrix, thres)
                self.all_metric_mapping[pred_pointer] = eval_metric

    def predict_pred_pointer(self, logit, pred_pointer, post_process=True):
        features_dict = self.pred_pointer_to_features_dict[pred_pointer]

        missing_labels = self.get_missing_labels(pred_pointer)

        if missing_labels:
            assert all(label in features_dict.keys() for label
                       in missing_labels)
        else:
            missing_labels = features_dict.keys()

        num_labels = len(missing_labels)
        num_candidates = features_dict.values()[0].shape[0]

        score_matrix = np.ndarray(
            shape=(num_labels, num_candidates))

        for row_idx, arg_label in enumerate(missing_labels):
            features = features_dict[arg_label]
            scores = logit.predict_proba(features)[:, 1]
            score_matrix[row_idx, :] = np.array(scores)

        if post_process:
            for column_idx in range(num_candidates):
                max_score_idx = score_matrix[:, column_idx].argmax()
                for row_idx in range(num_labels):
                    if row_idx != max_score_idx:
                        score_matrix[row_idx, column_idx] = -1.0

        return score_matrix

    def predict_fold(self, logit, fold_idx, post_process=False):
        score_matrix_mapping = {}
        for pred_pointer in self.fold_idx_to_pred_pointer[fold_idx]:
            score_matrix = self.predict_pred_pointer(
                logit, pred_pointer, post_process=post_process)
            score_matrix_mapping[pred_pointer] = score_matrix

        return score_matrix_mapping

    def search_threshold(self, score_matrix_mapping):
        thres_list = [float(x) / 100 for x in range(0, 100)]
        best_thres = -1
        best_f1 = 0
        for thres in thres_list:
            f1 = self.eval_multi(score_matrix_mapping, thres).f1()
            if f1 > best_f1:
                best_thres = thres
                best_f1 = f1

        eval_metric = self.eval_multi(score_matrix_mapping, best_thres)
        return best_thres, eval_metric

    def eval_pred_pointer(self, pred_pointer, score_matrix, thres):
        num_gold, dice_score_dict = self.dice_score_dict_mapping[pred_pointer]
        missing_labels = self.get_missing_labels(pred_pointer)
        return DiceEvalMetric.eval(
            num_gold, dice_score_dict, score_matrix, thres=thres,
            missing_labels=missing_labels)

    def eval_multi(self, score_matrix_mapping, thres):
        eval_metric = DiceEvalMetric()

        for pred_pointer, score_matrix in score_matrix_mapping.items():
            eval_metric.add_metric(
                self.eval_pred_pointer(pred_pointer, score_matrix, thres))

        return eval_metric

    def print_stats(self, fout=None):
        all_metric = DiceEvalMetric()
        metric_by_pred = defaultdict(DiceEvalMetric)
        metric_by_fold = defaultdict(DiceEvalMetric)

        for pred_pointer, eval_metric in self.all_metric_mapping.items():
            all_metric.add_metric(eval_metric)

            n_pred = self.pred_pointer_to_n_pred[pred_pointer]
            metric_by_pred[n_pred].add_metric(eval_metric)

            fold_idx = self.pred_pointer_to_fold_idx[pred_pointer]
            metric_by_fold[fold_idx].add_metric(eval_metric)

        log.info('=' * 20)
        log.info('All: ' + all_metric.to_text())

        pred_table_content = []
        for n_pred, eval_metric in metric_by_pred.items():
            table_row = \
                [n_pred, int(eval_metric.num_gold), eval_metric.precision(),
                 eval_metric.recall(), eval_metric.f1()]
            pred_table_content.append(table_row)

        pred_table_content.sort(key=itemgetter(1), reverse=True)

        table_row = \
            ['overall', int(all_metric.num_gold), all_metric.precision(),
             all_metric.recall(), all_metric.f1()]

        pred_table_content.append([''] * len(table_row))
        pred_table_content.append(table_row)

        pred_table_header = ['predicate', '# iarg', 'precision', 'recall', 'f1']

        pred_table = Texttable()
        pred_table.set_deco(Texttable.BORDER | Texttable.HEADER)
        pred_table.set_cols_align(['c'] * len(pred_table_header))
        pred_table.set_cols_valign(['m'] * len(pred_table_header))
        pred_table.set_cols_width([15] * len(pred_table_header))
        pred_table.set_precision(2)

        pred_table.header(pred_table_header)
        for row in pred_table_content:
            pred_table.add_row(row)

        fold_table_content = []
        tune_w = False
        for fold_idx, eval_metric in metric_by_fold.items():
            table_row = [fold_idx, int(eval_metric.num_gold)]

            model_params = self.model_state_list[fold_idx].logit.get_params()
            table_row.append(model_params['C'])
            if model_params['class_weight'] != 'balanced':
                table_row.append(model_params['class_weight'][1])
                tune_w = True
            table_row.append(self.model_state_list[fold_idx].thres)

            table_row.append(eval_metric.precision())
            table_row.append(eval_metric.recall())
            table_row.append(eval_metric.f1())
            fold_table_content.append(table_row)

        fold_table_header = ['fold', '# iarg', 'c']
        if tune_w:
            fold_table_header.append('w+')
        fold_table_header.extend(['t', 'precision', 'recall', 'f1'])

        fold_table = Texttable()
        fold_table.set_deco(Texttable.BORDER | Texttable.HEADER)
        fold_table.set_cols_align(['c'] * len(fold_table_header))
        fold_table.set_cols_valign(['m'] * len(fold_table_header))
        fold_table.set_cols_width([10] * len(fold_table_header))
        fold_table.set_precision(2)

        fold_table.header(fold_table_header)
        for row in fold_table_content:
            fold_table.add_row(row)

        print pred_table.draw()
        print
        print fold_table.draw()

        if fout:
            fout.write(pred_table.draw())
            fout.write('\n\n')
            fout.write(fold_table.draw())
            fout.close()
