import pickle as pkl
import timeit
from collections import OrderedDict
from operator import itemgetter

from common.imp_arg import helper
from common.imp_arg.candidate import Candidate
from common.imp_arg.tree_pointer import TreePointer
from utils import check_type, get_console_logger

log = get_console_logger()


class CandidateDict(object):
    def __init__(self, propbank_reader=None, nombank_reader=None, max_dist=2):
        self._propbank_reader = propbank_reader
        self._nombank_reader = nombank_reader
        self._max_dist = max_dist
        self._candidate_dict = OrderedDict()

    def __iter__(self):
        for key, candidates in self._candidate_dict.items():
            yield key, candidates

    @property
    def read_only(self):
        if self._propbank_reader and self._nombank_reader:
            return False
        else:
            return True

    def get_candidates(self, pred_pointer):
        check_type(pred_pointer, TreePointer)

        fileid = pred_pointer.fileid
        candidates = []

        # get all candidates from sentnum - max_dist to sentnum - 1
        for sentnum in range(max(0, pred_pointer.sentnum - self._max_dist),
                             pred_pointer.sentnum):
            key = '{}:{}'.format(fileid, sentnum)
            assert key in self._candidate_dict
            candidates.extend(self._candidate_dict[key])

        # for candidates within the same sentence, only add candidate
        # if it doesn't contain the pred_node in it's pred_pointer_list
        key = '{}:{}'.format(fileid, pred_pointer.sentnum)
        assert key in self._candidate_dict
        for candidate in self._candidate_dict[key]:
            if pred_pointer not in map(
                    itemgetter(0), candidate.pred_pointer_list):
                candidates.append(candidate)

        return candidates

    def add_candidates(self, fileid, sentnum):
        assert not self.read_only, 'cannot add candidates in read_only mode'

        instances = []
        instances.extend(self._propbank_reader.search_by_fileid(fileid))
        instances.extend(self._nombank_reader.search_by_fileid(fileid))

        for sentnum in range(max(0, sentnum - self._max_dist), sentnum + 1):
            key = '{}:{}'.format(fileid, sentnum)
            if key not in self._candidate_dict:
                self.add_key(key, instances)

    def add_key(self, key, instances):
        assert not self.read_only

        assert key not in self._candidate_dict

        fileid = key.split(':')[0]
        sentnum = int(key.split(':')[1])

        candidate_list = []
        arg_pointer_list = []

        for instance in instances:
            assert helper.shorten_wsj_fileid(instance.fileid) == fileid

            if instance.sentnum == sentnum:

                for candidate in Candidate.from_instance(instance):
                    if candidate.arg_pointer not in arg_pointer_list:
                        # filter candidate with zero non-trace tokens
                        if candidate.arg_pointer.flat_idx_list(no_trace=True):
                            arg_pointer_list.append(candidate.arg_pointer)
                            candidate_list.append(candidate)

                    else:
                        index = arg_pointer_list.index(candidate.arg_pointer)
                        candidate_list[index].merge(candidate)

        self._candidate_dict[key] = candidate_list

    def parse_corenlp(self, corenlp_mapping):
        log.info('Parsing CoreNLP information for all candidates')
        for key in self._candidate_dict:
            fileid = key.split(':')[0]
            sentnum = int(key.split(':')[1])

            idx_mapping, doc = corenlp_mapping[fileid]
            corenlp_sent = doc.get_sent(sentnum)
            sent_idx_mapping = idx_mapping[sentnum]

            for candidate in self._candidate_dict[key]:
                candidate.parse_corenlp(corenlp_sent, sent_idx_mapping)

    @classmethod
    def load(cls, candidate_dict_path, propbank_reader=None,
             nombank_reader=None, max_dist=2):
        log.info('Loading candidate dict from {}'.format(candidate_dict_path))

        start_time = timeit.default_timer()

        result = cls(
            propbank_reader=propbank_reader,
            nombank_reader=nombank_reader,
            max_dist=max_dist)

        candidate_dict = pkl.load(open(candidate_dict_path, 'r'))
        result._candidate_dict = candidate_dict

        elapsed = timeit.default_timer() - start_time
        log.info('Done in {:.3f} seconds'.format(elapsed))

        return result

    def save(self, candidate_dict_path):
        log.info('Saving candidate dict to {}'.format(candidate_dict_path))
        pkl.dump(self._candidate_dict, open(candidate_dict_path, 'w'))
