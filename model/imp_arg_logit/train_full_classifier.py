import argparse
from os import makedirs
from os.path import exists, join

from joblib import Parallel, delayed

from common.imp_arg import ImplicitArgumentDataset
from config import cfg
from model.imp_arg_logit.classifier import FullClassifier
from model.imp_arg_logit.helper import global_train
from utils import add_file_handler, log

parser = argparse.ArgumentParser()
parser.add_argument('--use_list', action='store_true',
                    help='if turned on, the p_1_word / p_2_word / p_3_word '
                         'features would be created as a list rather than '
                         'concatenation')
parser.add_argument('--featurizer', default='one_hot',
                    help='the featurizer to transform all features,'
                         'one_hot (default) or hash')
parser.add_argument('--fit_intercept', action='store_true',
                    help='if turned on, the logistic regression model '
                         'would add a constant intercept (1) to the '
                         'decision function')
parser.add_argument('--tune_w', action='store_true',
                    help='if turned on, the logistic regression model '
                         'would search for best class_weight parameter, '
                         'otherwise class_weight is set to balanced')
parser.add_argument('--missing_labels_path',
                    help='path to missing labels mapping, if not provided, '
                         'use default missing labels')
parser.add_argument('--use_val', action='store_true',
                    help='if turned on, in cross validation a separate '
                         'fold would be used as validation set, otherwise '
                         'all training folds would be used for validation')
parser.add_argument('--verbose', action='store_true',
                    help='if turned on, evaluation results on every '
                         'parameter grid on every fold would be printed')
parser.add_argument('--log_to_file', action='store_true',
                    help='if turned on, logs would be written to file')
parser.add_argument('--save_results', action='store_true',
                    help='if turned on, results would be written to file')
parser.add_argument('--save_models', action='store_true',
                    help='if turned on, all trained models would be saved')
parser.add_argument('--n_jobs', default=1, type=int,
                    help='number of parallel jobs, default = 1')

args = parser.parse_args()

if args.use_list:
    path_prefix = join(
        cfg.data_path, 'imp_arg_logit', 'full_classifier', 'list_features')
else:
    path_prefix = join(
        cfg.data_path, 'imp_arg_logit', 'full_classifier', 'concat_features')

if not exists(path_prefix):
    makedirs(path_prefix)

suffix = '{}-{}-{}-{}'.format(
    args.featurizer,
    'intercept' if args.fit_intercept else 'no_intercept',
    'tune_weight' if args.tune_w else 'balanced_weight',
    'use_val' if args.use_val else 'use_train')

if args.log_to_file:
    log_file_path = join(path_prefix, 'log-{}.log'.format(suffix))

    add_file_handler(log, log_file_path)

classifier = FullClassifier(n_splits=10)

dataset = ImplicitArgumentDataset()
classifier.read_dataset(dataset)

if args.missing_labels_path:
    classifier.load_missing_labels_mapping(args.missing_labels_path)

sample_list_path = join(path_prefix, 'sample_list.pkl')
if exists(sample_list_path):
    classifier.load_sample_list(sample_list_path)
else:
    classifier.build_sample_list(
        use_list=args.use_list, save_path=sample_list_path)

classifier.index_sample_list()

classifier.preprocess_features(args.featurizer)

classifier.index_raw_features()

classifier.set_hyper_parameter(
    fit_intercept=args.fit_intercept, tune_w=args.tune_w)

model_state_list = \
    Parallel(n_jobs=args.n_jobs, verbose=10, backend='multiprocessing')(
        delayed(global_train)(
            classifier, test_fold_idx,
            use_val=args.use_val, verbose=args.verbose, log_to_file=False,
            log_file_path=join(path_prefix, 'log-{}'.format(suffix)))
        for test_fold_idx in range(classifier.n_splits))

classifier.set_model_states(model_state_list)

if args.save_models:
    model_save_path = join(path_prefix, 'model-{}.pkl'.format(suffix))
    classifier.save_models(model_save_path)

fout_results = None
if args.save_results:
    results_path = join(path_prefix, 'results-{}.txt'.format(suffix))
    fout_results = open(results_path, 'w')

classifier.print_stats(fout=fout_results)
