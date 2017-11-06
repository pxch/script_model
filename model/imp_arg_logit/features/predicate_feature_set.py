from collections import OrderedDict

from nltk.tree import Tree

from common.corenlp import Document
from common.imp_arg.proposition import Proposition
from model.imp_arg_logit import helper
from model.imp_arg_logit.features.feature_set import FeatureSet
from utils import check_type, log

predicate_feature_list = [
    # features only related to p
    'p', 'p_p_synset', 'p_p_syn_cat', 'p_p_suffix',
    # features related to p and its argument types
    'nom_p_iarg', 'ver_p_iarg', 'nom_p_all_exp_args',
    # features related to the frequency of p
    'freq_nom_p', 'freq_ver_p', 'avg_freq_nom_p', 'avg_freq_ver_p',
    # features related to the context window of p
    'p_1_word', 'p_2_word', 'p_3_word',
    # features related to the left sibling of p
    'p_l_sib_syn_head', 'p_l_sib_first', 'p_l_sib_len',
    'p_l_sib_quantifier?',
    # features related to the right sibling of p
    'p_r_sib_syn_head', 'p_r_sib_first', 'p_r_sib_len',
    'p_r_sib_last', 'p_r_sib_head_pos', 'p_r_sib_syn_cat',
    # features related to the parent of p
    'p_par_grammar', 'p_par_head', 'p_par_head_pos', 'p_par_syn_cat',
    'p_head_of_par?',
    # features related to the syntactic structure of p
    'p_before_pass_verb?', 'p_follow_by_pp?',
    'p_following_pp_obj_head', 'p_syn_tree_path_to_nearest_pass_verb',
]


class PredicateFeatureSet(FeatureSet):
    def set_imp_arg(self, iarg_type):
        self._feature_map['nom_p_iarg'] += '-' + iarg_type
        self._feature_map['ver_p_iarg'] += '-' + iarg_type

    @classmethod
    def build(cls, proposition, doc, idx_mapping, use_list):
        check_type(proposition, Proposition)
        check_type(doc, Document)

        n_pred = proposition.n_pred
        v_pred = proposition.v_pred
        pred_pointer = proposition.pred_pointer
        sentnum = pred_pointer.sentnum
        wordnum = pred_pointer.tree_pointer.wordnum
        mapped_wordnum = idx_mapping[sentnum].index(wordnum)

        tree = doc.sents[sentnum].tree
        check_type(tree, Tree)

        core_arg_mapping = helper.predicate_core_arg_mapping[v_pred]

        lemma_list_per_sent = \
            [[token.lemma for token in sent.tokens] for sent in doc.sents]
        lemma_list = \
            [lemma for sub_list in lemma_list_per_sent for lemma in sub_list]

        sent = doc.get_sent(sentnum)

        pred_treepos = helper.get_treepos(tree, mapped_wordnum)

        feature_map = OrderedDict()

        # features only related to p
        feature_map['p'] = n_pred
        feature_map['p_p_synset'] = '-'.join(
            [n_pred, helper.nominal_predicate_synset_mapping[n_pred]])
        feature_map['p_p_syn_cat'] = '-'.join(
            [n_pred, tree[pred_treepos].label()])
        feature_map['p_p_suffix'] = '-'.join(
            [n_pred, helper.nominal_predicate_suffix_mapping[n_pred]])

        # features related to p and its argument types
        # set later by different imp_arg_type
        feature_map['nom_p_iarg'] = n_pred
        feature_map['ver_p_iarg'] = v_pred
        feature_map['nom_p_all_exp_args'] = '-'.join(
            [n_pred] +
            [core_arg_mapping[arg_label] for arg_label
             in sorted(proposition.exp_args.keys())
             if arg_label in core_arg_mapping])

        # features related to the frequency of p
        feature_map['freq_nom_p'] = lemma_list.count(n_pred)
        feature_map['freq_ver_p'] = lemma_list.count(v_pred)
        feature_map['avg_freq_nom_p'] = \
            float(sum([sub_list.count(n_pred) for sub_list
                       in lemma_list_per_sent])) / doc.num_sents
        feature_map['avg_freq_ver_p'] = \
            float(sum([sub_list.count(v_pred) for sub_list
                       in lemma_list_per_sent])) / doc.num_sents

        # features related to the context window of p
        # TODO: should it be concatenated by dash, or just a list?
        if use_list:
            feature_map['p_1_word'] = [
                '-'.join([n_pred, sent.get_token(idx).lemma]) for idx
                in range(max(0, mapped_wordnum - 1),
                         min(sent.num_tokens, mapped_wordnum + 2))
                if idx != mapped_wordnum]
            feature_map['p_2_word'] = [
                '-'.join([n_pred, sent.get_token(idx).lemma]) for idx
                in range(max(0, mapped_wordnum - 2),
                         min(sent.num_tokens, mapped_wordnum + 3))
                if idx != mapped_wordnum]
            feature_map['p_3_word'] = [
                '-'.join([n_pred, sent.get_token(idx).lemma]) for idx
                in range(max(0, mapped_wordnum - 3),
                         min(sent.num_tokens, mapped_wordnum + 4))
                if idx != mapped_wordnum]
        else:
            feature_map['p_1_word'] = '-'.join(
                [sent.get_token(idx).lemma for idx in
                 range(max(0, mapped_wordnum - 1),
                       min(sent.num_tokens, mapped_wordnum + 2))])
            feature_map['p_2_word'] = '-'.join(
                [sent.get_token(idx).lemma for idx in
                 range(max(0, mapped_wordnum - 2),
                       min(sent.num_tokens, mapped_wordnum + 3))])
            feature_map['p_3_word'] = '-'.join(
                [sent.get_token(idx).lemma for idx in
                 range(max(0, mapped_wordnum - 3),
                       min(sent.num_tokens, mapped_wordnum + 4))])

        # features related to the left sibling of p
        l_sibling_treepos = helper.get_left_sibling(pred_treepos)
        if l_sibling_treepos is not None:
            l_sibling_range = \
                helper.get_token_range(tree, l_sibling_treepos)
            assert len(l_sibling_range) > 0, 'invalid l_sibling_treepos'

            # TODO: should I get syntactic head by dependency parse?
            l_sibling_head_idx = sent.dep_graph.get_head_token_idx(
                l_sibling_range[0], l_sibling_range[-1] + 1,
                msg_prefix=pred_pointer.fileid)
            feature_map['p_l_sib_syn_head'] = '-'.join(
                [n_pred, sent.get_token(l_sibling_head_idx).lemma])
            # TODO: should I use the lemma form?
            feature_map['p_l_sib_first'] = '-'.join(
                [n_pred, sent.get_token(l_sibling_range[0]).lemma])
            feature_map['p_l_sib_len'] = '-'.join(
                map(str, [n_pred, len(l_sibling_range)]))

            # TODO: should I include predicate in this feature?
            l_quantifier_flag = 0
            l_sibling_lemma_string = ' '.join(
                [sent.get_token(idx).lemma for idx in l_sibling_range])
            for quantifier in helper.quantifier_list:
                if quantifier in l_sibling_lemma_string:
                    l_quantifier_flag = 1
                    break
            feature_map['p_l_sib_quantifier?'] = l_quantifier_flag

        else:
            log.warning(
                'No left sibling found for predicate: {}'.format(pred_pointer))
            # TODO: should I add any empty / fake left sibling info?
            feature_map['p_l_sib_syn_head'] = n_pred
            feature_map['p_l_sib_first'] = n_pred
            feature_map['p_l_sib_len'] = n_pred

            feature_map['p_l_sib_quantifier?'] = 0

        # features related to the right sibling of p
        r_sibling_treepos = helper.get_right_sibling(tree, pred_treepos)
        if r_sibling_treepos is not None:
            r_sibling_range = \
                helper.get_token_range(tree, r_sibling_treepos)
            assert len(r_sibling_range) > 0, 'invalid r_sibling_treepos'

            r_sibling_head_idx = sent.dep_graph.get_head_token_idx(
                r_sibling_range[0], r_sibling_range[-1] + 1,
                msg_prefix=pred_pointer.fileid)
            feature_map['p_r_sib_syn_head'] = '-'.join(
                [n_pred, sent.get_token(r_sibling_head_idx).lemma])
            feature_map['p_r_sib_first'] = '-'.join(
                [n_pred, sent.get_token(r_sibling_range[0]).lemma])
            feature_map['p_r_sib_len'] = '-'.join(
                map(str, [n_pred, len(r_sibling_range)]))

            feature_map['p_r_sib_last'] = '-'.join(
                [n_pred, sent.get_token(r_sibling_range[-1]).lemma])
            feature_map['p_r_sib_head_pos'] = '-'.join(
                [n_pred, sent.get_token(r_sibling_head_idx).pos])
            feature_map['p_r_sib_syn_cat'] = '-'.join(
                [n_pred, tree[r_sibling_treepos].label()])

        else:
            log.warning(
                'No right sibling found for predicate: {}'.format(pred_pointer))
            feature_map['p_r_sib_syn_head'] = n_pred
            feature_map['p_r_sib_first'] = n_pred
            feature_map['p_r_sib_len'] = n_pred

            feature_map['p_r_sib_last'] = n_pred
            feature_map['p_r_sib_head_pos'] = n_pred
            feature_map['p_r_sib_syn_cat'] = n_pred

        # features related to the parent of p
        par_treepos = helper.get_parent(tree, pred_treepos)
        assert par_treepos is not None, 'parent treepos should never be None'

        par_grammar = tree[par_treepos].label()
        for child in tree[par_treepos]:
            par_grammar += '_' + child.label()
        feature_map['p_par_grammar'] = '-'.join([n_pred, par_grammar])

        par_range = helper.get_token_range(tree, par_treepos)
        assert len(par_range) > 0, 'invalid par_treepos'

        par_head_idx = sent.dep_graph.get_head_token_idx(
            par_range[0], par_range[-1] + 1, msg_prefix=pred_pointer.fileid)
        feature_map['p_par_head'] = '-'.join(
            [n_pred, sent.get_token(par_head_idx).lemma])
        feature_map['p_par_head_pos'] = '-'.join(
            [n_pred, sent.get_token(par_head_idx).pos])
        feature_map['p_par_syn_cat'] = '-'.join(
            [n_pred, tree[par_treepos].label()])

        head_of_par_flag = (par_head_idx == mapped_wordnum)
        feature_map['p_head_of_par?'] = '-'.join(
            map(str, [n_pred, head_of_par_flag]))

        # features related to the syntactic structure of p
        before_pass_verb_flag = 0
        if len(sent.lookup_label('mod', mapped_wordnum, 'nsubjpass')) > 0:
            before_pass_verb_flag = 1
        feature_map['p_before_pass_verb?'] = '-'.join(
            map(str, [n_pred, before_pass_verb_flag]))

        follow_by_pp_flag = 0
        pp_obj_idx = -1
        for label, indices in sent.dep_graph.lookup(
                'head', mapped_wordnum).items():
            if label.startswith('nmod:') and label not in \
                    ['nmod:agent', 'nmod:npmod', 'nmod:poss', 'nmod:tmod']:
                follow_by_pp_flag = 1
                pp_obj_idx = indices[0]
                break
        feature_map['p_follow_by_pp?'] = '-'.join(
            map(str, [n_pred, follow_by_pp_flag]))

        if pp_obj_idx != -1:
            pp_obj_head = sent.get_token(pp_obj_idx).lemma
            feature_map['p_following_pp_obj_head'] = \
                '-'.join([n_pred, pp_obj_head])
        else:
            feature_map['p_following_pp_obj_head'] = n_pred

        pass_verb_idx_list = []
        for idx, token, in enumerate(sent.tokens):
            if token.pos.startswith('VB') and \
                    sent.lookup_label('head', idx, 'nsubjpass'):
                pass_verb_idx_list.append(idx)

        # TODO: should I use the path from dependency parse?
        if pass_verb_idx_list:
            pass_verb_dist_list = \
                [abs(idx - mapped_wordnum) for idx in pass_verb_idx_list]
            nearest_pass_verb_idx = pass_verb_idx_list[
                pass_verb_dist_list.index(min(pass_verb_dist_list))]

            syn_tree_path = helper.get_syn_tree_path(
                sent.dep_graph, mapped_wordnum, nearest_pass_verb_idx,
                msg_prefix=pred_pointer.fileid)

            feature_map['p_syn_tree_path_to_nearest_pass_verb'] = \
                '_'.join(syn_tree_path)
        else:
            feature_map['p_syn_tree_path_to_nearest_pass_verb'] = ''

        predicate_feature_set = cls(feature_map)
        assert predicate_feature_set.feature_list == predicate_feature_list

        return predicate_feature_set
