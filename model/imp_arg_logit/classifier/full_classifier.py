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
from utils import check_type, log
from .base_classifier import BaseClassifier


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


class FullModelState(object):
    def __init__(self, logit, thres, val_fold_indices, val_metric,
                 test_fold_idx, test_metric):
        self.logit = logit
        self.thres = thres
        self.val_fold_indices = val_fold_indices
        self.val_metric = val_metric
        self.test_fold_idx = test_fold_idx
        self.test_metric = test_metric


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

        # list of test_score_matrix_mapping and best_thres for each fold
        self.all_score_matrix_mapping = []

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

    def train_model(self, test_fold_idx, use_val=False, verbose=False):
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
        log.info('=' * 20)
        log.info(
            'Test fold #{}, use fold #{} as validation'.format(
                test_fold_idx, val_fold_indices))

        train_features = self.features[train_sample_indices]
        train_gold = self.labels[train_sample_indices]

        best_param = None
        best_thres = -1
        best_val_f1 = 0
        best_val_metric = None

        logit = LogisticRegression()

        for param in self.param_grid:
            logit.set_params(**param)

            logit.fit(train_features, train_gold)

            val_score_matrix_mapping = {}
            for val_fold_idx in val_fold_indices:
                val_score_matrix_mapping.update(
                    self.predict_fold(logit, val_fold_idx, post_process=True))

            thres, val_metric = \
                self.search_threshold(val_score_matrix_mapping)

            debug_msg = \
                'Validation fold #{}, params = {}, thres = {:.2f}, {}'.format(
                    val_fold_indices, param, thres, val_metric.to_text())

            if verbose:
                log.info(debug_msg)
            else:
                log.debug(debug_msg)

            if val_metric.f1() > best_val_f1:
                best_param = param
                best_thres = thres
                best_val_f1 = val_metric.f1()
                best_val_metric = val_metric

        log.info('-' * 20)
        log.info(
            'Validation ford #{}, selecting best param = {}, thres = {:.2f} '
            'with validation f1 = {}'.format(
                val_fold_indices, best_param, best_thres, best_val_f1))

        logit.set_params(**best_param)
        logit.fit(train_features, train_gold)

        test_score_matrix_mapping = \
            self.predict_fold(logit, test_fold_idx, post_process=True)

        test_metric = self.eval(test_score_matrix_mapping, best_thres)

        log.info(
            'Test fold #{}, params = {}, thres = {:.2f}, {}'.format(
                test_fold_idx, best_param, best_thres, test_metric.to_text()))

        model_state = FullModelState(
            logit, best_thres,
            val_fold_indices, best_val_metric,
            test_fold_idx, test_metric)

        return model_state, test_score_matrix_mapping, best_thres

    def set_states(self, states):
        self.model_list = []
        self.all_score_matrix_mapping = []
        for model_state, test_score_matrix_mapping, best_thres in states:
            self.model_list.append(model_state)
            self.all_score_matrix_mapping.append(
                (test_score_matrix_mapping, best_thres))

    def predict_pred_pointer(self, logit, pred_pointer, post_process=False):
        sample_idx_dict = \
            self.pred_pointer_to_sample_idx_dict[pred_pointer]

        missing_labels = self.get_missing_labels(pred_pointer)

        if missing_labels:
            assert all(label in sample_idx_dict.keys() for label
                       in missing_labels)
        else:
            missing_labels = sample_idx_dict.keys()

        num_labels = len(missing_labels)
        num_candidates = len(sample_idx_dict.values()[0])

        score_matrix = np.ndarray(
            shape=(num_labels, num_candidates))

        for row_idx, arg_label in enumerate(missing_labels):
            sample_idx_list = sample_idx_dict[arg_label]
            scores = logit.predict_proba(
                self.features[sample_idx_list])[:, 1]
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
            f1 = self.eval(score_matrix_mapping, thres).f1()
            if f1 > best_f1:
                best_thres = thres
                best_f1 = f1

        eval_metric = self.eval(score_matrix_mapping, best_thres)
        return best_thres, eval_metric

    def eval_pred_pointer(self, pred_pointer, score_matrix, thres):
        num_gold, dice_score_dict = self.dice_score_dict_mapping[pred_pointer]
        missing_labels = self.get_missing_labels(pred_pointer)
        return DiceEvalMetric.eval(
            num_gold, dice_score_dict, score_matrix, thres=thres,
            missing_labels=missing_labels)

    def eval(self, score_matrix_mapping, thres):
        eval_metric = DiceEvalMetric()

        for pred_pointer, score_matrix in score_matrix_mapping.items():
            eval_metric.add_metric(
                self.eval_pred_pointer(pred_pointer, score_matrix, thres))

        return eval_metric

    def print_stats(self, fout=None):
        all_metric = DiceEvalMetric()
        metric_by_pred = defaultdict(DiceEvalMetric)
        metric_by_fold = defaultdict(DiceEvalMetric)

        for score_matrix_mapping, thres in self.all_score_matrix_mapping:
            for pred_pointer, score_matrix in score_matrix_mapping.items():
                eval_metric = self.eval_pred_pointer(
                    pred_pointer, score_matrix, thres)

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

            model_params = self.model_list[fold_idx].logit.get_params()
            table_row.append(model_params['C'])
            if model_params['class_weight'] != 'balanced':
                table_row.append(model_params['class_weight'][1])
                tune_w = True
            table_row.append(self.model_list[fold_idx].thres)

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
