import pickle as pkl
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter

from sklearn.linear_model import LogisticRegression
from texttable import Texttable

from common.imp_arg import ImplicitArgumentDataset
from common.imp_arg import helper
from model.imp_arg_logit.features import PredicateFeatureSet
from utils import BaseEvalMetric
from utils import check_type, log
from .base_classifier import BaseClassifier


class BinarySample(object):
    def __init__(self, pred_pointer, fold_idx, arg_label, feature_set, label):
        self.pred_pointer = pred_pointer
        self.fold_idx = fold_idx
        self.arg_label = arg_label
        self.feature_set = feature_set
        self.label = label

    @classmethod
    def from_proposition(cls, proposition, corenlp_mapping, fold_idx,
                         use_list=False):
        sample_list = []
        pred_pointer = proposition.pred_pointer
        idx_mapping, doc = corenlp_mapping[pred_pointer.fileid]

        predicate_feature_set = PredicateFeatureSet.build(
            proposition, doc, idx_mapping, use_list)

        missing_labels = proposition.missing_labels()

        core_arg_mapping = helper.predicate_core_arg_mapping[proposition.v_pred]

        for arg_label in missing_labels:
            feature_set = deepcopy(predicate_feature_set)

            iarg_type = core_arg_mapping[arg_label]
            feature_set.set_imp_arg(iarg_type)

            if arg_label in proposition.imp_args:
                label = 1
            else:
                label = 0

            sample_list.append(cls(
                str(pred_pointer), fold_idx, arg_label, feature_set, label))
        return sample_list


class BinaryModelState(object):
    def __init__(self, logit, val_fold_indices, val_stats, test_fold_idx,
                 test_stats):
        self.logit = logit
        self.val_fold_indices = val_fold_indices
        self.val_stats = val_stats
        self.test_fold_idx = test_fold_idx
        self.test_stats = test_stats


class BinaryEvalMetric(BaseEvalMetric):
    def __init__(self, tp=0, fp=0, fn=0, tn=0):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.num_all = tp + fp + fn + tn
        self.num_correct = tp + tn
        self.num_positive = tp + fn

    def add_metric(self, other):
        assert isinstance(other, BinaryEvalMetric)
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        self.num_all += other.num_all
        self.num_correct += other.num_correct
        self.num_positive += other.num_positive

    def precision(self):
        if self.tp + self.fp > 0:
            return 100. * self.tp / (self.tp + self.fp)
        else:
            return 0.

    def recall(self):
        if self.num_positive > 0:
            return 100. * self.tp / self.num_positive
        else:
            return 0.

    def accuracy(self):
        return 100. * self.num_correct / self.num_all

    def to_text(self):
        return 'accuracy = {} / {} = {:6.2f} %, {}'.format(
            self.num_correct, self.num_all, self.accuracy(), str(self))

    @classmethod
    def eval(cls, predict, gold):
        assert len(predict) == len(gold)
        tp = sum([1 for y1, y2 in zip(predict, gold) if y1 == 1 and y2 == 1])
        fp = sum([1 for y1, y2 in zip(predict, gold) if y1 == 1 and y2 == 0])
        fn = sum([1 for y1, y2 in zip(predict, gold) if y1 == 0 and y2 == 1])
        tn = sum([1 for y1, y2 in zip(predict, gold) if y1 == 0 and y2 == 0])
        return cls(tp=tp, fp=fp, fn=fn, tn=tn)


class BinaryClassifier(BaseClassifier):
    def __init__(self, n_splits=10):
        super(BinaryClassifier, self).__init__(n_splits=n_splits)

        # mapping from pred_pointers to list of sample indices
        self.pred_pointer_to_sample_idx_list = None

        # mapping from pred_pointer and list of predicted labels
        self.all_prediction_mapping = None

        # mapping from pred_pointer to list of predicted missing labels
        self.missing_labels_mapping = None

    def read_dataset(self, dataset):
        check_type(dataset, ImplicitArgumentDataset)
        log.info('Reading implicit argument dataset')

        dataset.load_propositions(helper.propositions_path)

        self.dataset = dataset

        self.index_dataset()

    def build_sample_list(self, use_list=False, save_path=None):
        assert self.dataset is not None
        log.info('Building sample list from every missing argument label '
                 'of every proposition')

        corenlp_mapping = self.dataset.corenlp_mapping

        self.sample_list = []

        for proposition in self.dataset.propositions:
            fold_idx = \
                self.pred_pointer_to_fold_idx[str(proposition.pred_pointer)]
            sample_list = BinarySample.from_proposition(
                proposition, corenlp_mapping, fold_idx, use_list)
            self.sample_list.extend(sample_list)

        if save_path:
            log.info('Saving sample list to {}'.format(save_path))
            pkl.dump(self.sample_list, open(save_path, 'w'))

    def index_sample_list(self):
        super(BinaryClassifier, self).index_sample_list()

        self.pred_pointer_to_sample_idx_list = defaultdict(list)
        log.info('Building mapping from pred_pointer to list of sample indices')

        for sample_idx, sample in enumerate(self.sample_list):
            pred_pointer = sample.pred_pointer
            self.pred_pointer_to_sample_idx_list[pred_pointer].append(
                sample_idx)

    def reset_states(self):
        self.model_list = []
        self.all_prediction_mapping = {}

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

        val_sample_indices = \
            [sample_idx for sample_idx, fold_idx
             in enumerate(self.sample_idx_to_fold_idx)
             if fold_idx in val_fold_indices]

        val_features = self.features[val_sample_indices]
        val_gold = self.labels[val_sample_indices]

        best_param = None
        best_val_f1 = 0
        best_val_metric = None

        logit = LogisticRegression()

        for param in self.param_grid:
            logit.set_params(**param)

            logit.fit(train_features, train_gold)

            val_metric = BinaryEvalMetric.eval(
                logit.predict(val_features), val_gold)

            debug_msg = 'Validation fold #{}, params = {}, {}'.format(
                val_fold_indices, param, val_metric.to_text())

            if verbose:
                log.info(debug_msg)
            else:
                log.debug(debug_msg)

            if val_metric.f1() > best_val_f1:
                best_param = param
                best_val_f1 = val_metric.f1()
                best_val_metric = val_metric

        log.info('-' * 20)
        log.info(
            'Validation fold #{}, selecting best param = {}, '
            'with validation f1 = {}'.format(
                val_fold_indices, best_param, best_val_f1))

        logit.set_params(**best_param)
        logit.fit(train_features, train_gold)

        test_prediction_mapping = self.predict_fold(logit, test_fold_idx)

        test_metric = self.eval(test_prediction_mapping)

        log.info('Test fold #{}, params = {}, {}'.format(
            test_fold_idx, best_param, test_metric.to_text()))

        model_state = BinaryModelState(
            logit, val_fold_indices, best_val_metric,
            test_fold_idx, test_metric)

        return model_state, test_prediction_mapping

    def set_states(self, states):
        self.model_list = []
        self.all_prediction_mapping = {}
        for model_state, test_prediction_mapping in states:
            self.model_list.append(model_state)
            self.all_prediction_mapping.update(test_prediction_mapping)

    def predict_pred_pointer(self, logit, pred_pointer):
        sample_idx_list = self.pred_pointer_to_sample_idx_list[pred_pointer]
        return logit.predict(self.features[sample_idx_list])

    def predict_fold(self, logit, fold_idx):
        prediction_mapping = {}
        for pred_pointer in self.fold_idx_to_pred_pointer[fold_idx]:
            prediction_mapping[pred_pointer] = \
                self.predict_pred_pointer(logit, pred_pointer)
        return prediction_mapping

    def eval_pred_pointer(self, pred_pointer, prediction):
        sample_idx_list = self.pred_pointer_to_sample_idx_list[pred_pointer]
        gold = self.labels[sample_idx_list]
        return BinaryEvalMetric.eval(prediction, gold)

    def eval(self, prediction_mapping):
        eval_metric = BinaryEvalMetric()

        for pred_pointer, prediction in prediction_mapping.items():
            eval_metric.add_metric(
                self.eval_pred_pointer(pred_pointer, prediction))

        return eval_metric

    def predict_missing_labels(self, save_path=None):
        log.info('Predicting missing labels for all predicates')
        self.missing_labels_mapping = defaultdict(list)

        for pred_pointer, prediction in self.all_prediction_mapping.items():
            sample_idx_list = self.pred_pointer_to_sample_idx_list[pred_pointer]
            arg_label_list = \
                [self.sample_list[sample_idx].arg_label
                 for sample_idx in sample_idx_list]
            for label_pred, arg_label in zip(prediction, arg_label_list):
                if label_pred:
                    self.missing_labels_mapping[pred_pointer].append(arg_label)

        if save_path:
            log.info('Saving predicted missing labels to {}'.format(save_path))
            pkl.dump(self.missing_labels_mapping, open(save_path, 'w'))

    def print_stats(self, fout=None):
        all_metric = BinaryEvalMetric()
        metric_by_pred = defaultdict(BinaryEvalMetric)
        metric_by_fold = defaultdict(BinaryEvalMetric)

        for pred_pointer, prediction in self.all_prediction_mapping.items():
            eval_metric = self.eval_pred_pointer(pred_pointer, prediction)

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
                [n_pred, eval_metric.num_positive, eval_metric.num_all,
                 eval_metric.accuracy(), eval_metric.precision(),
                 eval_metric.recall(), eval_metric.f1()]
            pred_table_content.append(table_row)

        pred_table_content.sort(key=itemgetter(1), reverse=True)

        table_row = \
            ['overall', all_metric.num_positive, all_metric.num_all,
             all_metric.accuracy(), all_metric.precision(),
             all_metric.recall(), all_metric.f1()]

        pred_table_content.append([''] * len(table_row))
        pred_table_content.append(table_row)

        pred_table_header = ['predicate', '# iarg', '# missing',
                             'accuracy', 'precision', 'recall', 'f1']

        pred_table = Texttable()
        pred_table.set_deco(Texttable.BORDER | Texttable.HEADER)
        pred_table.set_cols_align(['c'] * len(pred_table_header))
        pred_table.set_cols_valign(['m'] * len(pred_table_header))
        pred_table.set_cols_width([10] * len(pred_table_header))
        pred_table.set_precision(2)

        pred_table.header(pred_table_header)
        for row in pred_table_content:
            pred_table.add_row(row)

        fold_table_content = []
        for fold_idx, eval_metric in metric_by_fold.items():
            table_row = \
                [fold_idx, eval_metric.num_positive, eval_metric.num_all,
                 eval_metric.accuracy(), eval_metric.precision(),
                 eval_metric.recall(), eval_metric.f1()]
            fold_table_content.append(table_row)

        table_row = \
            ['overall', all_metric.num_positive, all_metric.num_all,
             all_metric.accuracy(), all_metric.precision(),
             all_metric.recall(), all_metric.f1()]

        fold_table_content.append([''] * len(table_row))
        fold_table_content.append(table_row)

        fold_table_header = ['fold', '# iarg', '# missing',
                             'accuracy', 'precision', 'recall', 'f1']

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
