from collections import OrderedDict, defaultdict

from nltk.corpus import wordnet as wn
from nltk.tree import Tree

from common.corenlp import Coreference, Mention
from common.corenlp import Document
from common.imp_arg.proposition import Proposition
from common.imp_arg.tree_pointer import TreePointer
from model.imp_arg_logit import helper
from model.imp_arg_logit.features.base_feature_set import BaseFeatureSet
from utils import check_type


class FillerFeatureSet(BaseFeatureSet):
    feature_list = [
        # features on primary filler only
        'c_p_dist', 'c_syn_cat_ver_p_iarg', 'c_p_arg_to_same_pred?',
        # features on all fillers in the coreference chain
        'num_elem',
        # features on sentence distances
        'min_f_p_dist', 'max_f_p_dist', 'avg_f_p_dist',
        # features on the count and percentage of fillers in coreference
        'num_arg_to_other_pred', 'per_arg_to_other_pred',
        'per_cop_subj_p_obj', 'per_cop_obj_p_subj',
        'per_cop_subj', 'per_cop_obj',
        'per_indefinite_np', 'per_definite_np', 'per_quantified_np',
        'per_sentential_subj',
        # features on the head word of each filler
        'f_head_ver_p_iarg', 'f_head_synset_ver_p_iarg',
        # features on the predicate of each filler
        'pf_argf_p_iarg', 'pf_synset_argf_p_synset_iarg', 'argf_iarg_same?',
        # 'min_wup_sim_pf_p'
        # TODO: add corpus statistics features
    ]

    @classmethod
    def build(cls, proposition, doc, iarg, c_filler, use_list):
        check_type(proposition, Proposition)
        check_type(doc, Document)
        check_type(c_filler, TreePointer)

        n_pred = proposition.n_pred
        v_pred = proposition.v_pred

        p_pointer = proposition.pred_pointer
        p_sentnum = p_pointer.sentnum
        p_sent = doc.get_sent(p_sentnum)
        p_idx = p_pointer.tree_pointer.wordnum

        c_sentnum = c_filler.sentnum
        c_sent = doc.get_sent(c_sentnum)

        kwargs = OrderedDict()

        kwargs['c_p_dist'] = p_sentnum - c_sentnum

        c_idx_list = \
            [idx for corenlp in c_filler.corenlp_list
             for idx in corenlp.idx_list]
        assert len(c_idx_list) > 0

        c_head_idx = c_filler.head_piece_corenlp().head_idx

        c_treepos = c_sent.tree.treeposition_spanning_leaves(
            c_idx_list[0], c_idx_list[-1] + 1)

        c_subtree = c_sent.tree[c_treepos]
        if not isinstance(c_subtree, Tree):
            c_treepos = c_treepos[:-1]
            c_subtree = c_sent.tree[c_treepos]

        core_arg_mapping = helper.predicate_core_arg_mapping[v_pred]
        iarg_label = core_arg_mapping[iarg]

        kwargs['c_syn_cat_ver_p_iarg'] = '-'.join([
            c_subtree.label(), v_pred, iarg_label])

        # TODO: is it correct to just match c_head_idx?
        kwargs['c_p_arg_to_same_pred?'] = 0
        if c_sentnum == p_sentnum:
            for pred_idx, pred_token in enumerate(p_sent.tokens):
                if pred_token.pos.startswith('VB'):
                    # exclude "be" verbs
                    if pred_token.lemma == 'be':
                        continue
                    # exclude modifying verbs
                    if p_sent.dep_graph.lookup_label(
                            'head', pred_idx, 'xcomp'):
                        continue

                    arg_idx_list = []
                    for subj in p_sent.get_subj_list(pred_idx):
                        arg_idx_list.append(subj.token_idx)
                    for obj in p_sent.get_dobj_list(pred_idx):
                        arg_idx_list.append(obj.token_idx)
                    for _, pobj in p_sent.get_pobj_list(pred_idx):
                        arg_idx_list.append(pobj.token_idx)

                    if p_idx in arg_idx_list and c_head_idx in arg_idx_list:
                        kwargs['c_p_arg_to_same_pred?'] = 1
                        break

        # build the coreference chain for the primary filler
        if c_filler.entity_idx != -1:
            coreference = doc.get_coref(c_filler.entity_idx)
        else:
            sent_idx = c_filler.sentnum
            head_token_idx = c_head_idx
            rep = True
            text = c_filler.surface(no_trace=True)
            mention = Mention(
                sent_idx=sent_idx,
                start_token_idx=c_idx_list[0],
                end_token_idx=c_idx_list[-1] + 1,
                head_token_idx=head_token_idx,
                rep=rep,
                text=text)
            mention.add_token_info(c_sent.tokens)

            coreference = Coreference(0)
            coreference.add_mention(mention)

        num_fillers = coreference.num_mentions
        kwargs['num_elem'] = num_fillers

        # build a map from sentence indices to list of pred-arg pairs
        pred_arg_list_map = {}
        for mention in coreference.mentions:
            if mention.sent_idx not in pred_arg_list_map:
                pred_arg_list = []
                f_sent = doc.get_sent(mention.sent_idx)

                for pred_idx, pred_token in enumerate(f_sent.tokens):
                    if pred_token.pos.startswith('VB'):
                        # exclude "be" verbs
                        if pred_token.lemma == 'be':
                            continue
                        # exclude modifying verbs
                        if f_sent.dep_graph.lookup_label(
                                'head', pred_idx, 'xcomp'):
                            continue

                        for subj in f_sent.get_subj_list(pred_idx):
                            pred_arg_list.append((pred_token, subj, 'SUBJ'))
                        for obj in f_sent.get_dobj_list(pred_idx):
                            pred_arg_list.append((pred_token, obj, 'OBJ'))
                        for prep, pobj in f_sent.get_pobj_list(pred_idx):
                            pred_arg_list.append(
                                (pred_token, pobj, 'PREP_' + prep))

                pred_arg_list_map[mention.sent_idx] = pred_arg_list

        # build a map from a feature name to a list of filler features
        filler_features = defaultdict(list)
        for mention in coreference.mentions:
            f_sent = doc.get_sent(mention.sent_idx)

            # features for min_f_p_dist, max_f_p_dist, and avg_f_p_dist
            filler_features['f_p_dist'].append(
                abs(p_sentnum - mention.sent_idx))

            # features for per_cop_subj_p_obj, per_cop_obj_p_subj,
            # per_cop_subj, and per_cop_obj
            cop_subj_p_obj_flag = 0
            cop_obj_p_subj_flag = 0
            cop_subj_flag = 0
            cop_obj_flag = 0

            # TODO: should I consider other variants of cop relations?
            if mention.sent_idx == p_sentnum:
                dep_graph = p_sent.dep_graph
                f_idx = mention.head_token_idx

                cop_obj_list = dep_graph.lookup_label('mod', f_idx, 'nsubj')
                if len(cop_obj_list) > 0:
                    cop_obj_idx = cop_obj_list[0]
                    if dep_graph.lookup_label('head', cop_obj_idx, 'cop'):
                        cop_subj_flag = 1
                        if cop_obj_idx == p_idx:
                            cop_subj_p_obj_flag = 1

                if dep_graph.lookup_label('head', f_idx, 'cop'):
                    cop_obj_flag = 1
                    cop_subj_list = \
                        dep_graph.lookup_label('head', f_idx, 'nsubj')
                    if len(cop_subj_list):
                        cop_subj_idx = cop_subj_list[0]
                        if cop_subj_idx == p_idx:
                            cop_obj_p_subj_flag = 1

            filler_features['cop_subj_p_obj'].append(cop_subj_p_obj_flag)
            filler_features['cop_obj_p_subj'].append(cop_obj_p_subj_flag)
            filler_features['cop_subj'].append(cop_subj_flag)
            filler_features['cop_obj'].append(cop_obj_flag)

            # features for per_indefinite_np, per_definite_np,
            # and per_quantified_np
            # TODO: can indefinite / definite NP be figured by the first token?
            mention_lemma_list = [token.lemma for token in mention.tokens]

            if mention_lemma_list[0] in helper.indefinite_determiner_list:
                filler_features['indefinite_np'].append(1)
            else:
                filler_features['indefinite_np'].append(0)

            if mention_lemma_list[0] in helper.definite_determiner_list:
                filler_features['definite_np'].append(1)
            else:
                filler_features['definite_np'].append(0)

            quantified_np_flag = 0
            mention_lemma = ' '.join(mention_lemma_list)
            for quantifier in helper.quantifier_list:
                if quantifier in mention_lemma:
                    quantified_np_flag = 1
                    break
            filler_features['quantified_np'].append(quantified_np_flag)

            # features for per_sentential_subj
            # TODO: is this correct?

            mention_treepos = f_sent.tree.treeposition_spanning_leaves(
                mention.start_token_idx, mention.end_token_idx)

            mention_subtree = f_sent.tree[mention_treepos]
            if not isinstance(mention_subtree, Tree):
                mention_treepos = mention_treepos[:-1]
                mention_subtree = f_sent.tree[mention_treepos]
            if mention_subtree.label() == 'S':
                filler_features['sentential_subj'].append(1)
            else:
                filler_features['sentential_subj'].append(0)

            # features on f_head_ver_p_iarg, f_head_synset_ver_p_iarg
            head_lemma = mention.head_token.lemma
            filler_features['head_lemma'].append(head_lemma)

            head_pos = mention.head_token.pos
            if head_pos.startswith('NN'):
                pos = wn.NOUN
            elif head_pos.startswith('VB'):
                pos = wn.VERB
            elif head_pos.startswith('JJ'):
                pos = wn.ADJ
            elif head_pos.startswith('RB'):
                pos = wn.ADV
            else:
                pos = None
            head_synset_id = ''
            if pos:
                head_synsets = wn.synsets(head_lemma, pos=pos)
                if head_synsets:
                    head_synset_id = head_synsets[0].name()
            filler_features['head_synset_id'].append(head_synset_id)

            # features for num_arg_to_other_pred, per_arg_to_other_pred,
            # pf_argf_p_iarg, pf_synset_argf_p_iarg, argf_iarg_same?,
            # and min_wup_sim_pf_p
            pred_arg_list = pred_arg_list_map[mention.sent_idx]

            arg_to_other_pred_flag = 0
            pred_f_lemma = ''
            arg_f_label = ''
            for pred_token, arg_token, label in pred_arg_list:
                if arg_token.token_idx == mention.head_token_idx:
                    if arg_to_other_pred_flag == 0:
                        arg_to_other_pred_flag = 1
                        pred_f_lemma = pred_token.lemma
                        arg_f_label = label

            filler_features['arg_to_other_pred'].append(arg_to_other_pred_flag)
            filler_features['pred_f_lemma'].append(pred_f_lemma)
            filler_features['arg_f_label'].append(arg_f_label)

            pred_f_synset_id = ''
            if pred_f_lemma:
                pred_f_synsets = wn.synsets(pred_f_lemma, pos=wn.VERB)
                if pred_f_synsets:
                    pred_f_synset_id = pred_f_synsets[0].name()
            filler_features['pred_f_synset_id'].append(pred_f_synset_id)

        kwargs['min_f_p_dist'] = min(filler_features['f_p_dist'])
        kwargs['max_f_p_dist'] = max(filler_features['f_p_dist'])
        kwargs['avg_f_p_dist'] = \
            float(sum(filler_features['f_p_dist'])) / num_fillers

        kwargs['per_cop_subj_p_obj'] = \
            float(sum(filler_features['cop_subj_p_obj'])) / num_fillers
        kwargs['per_cop_obj_p_subj'] = \
            float(sum(filler_features['cop_obj_p_subj'])) / num_fillers
        kwargs['per_cop_subj'] = \
            float(sum(filler_features['cop_subj'])) / num_fillers
        kwargs['per_cop_obj'] = \
            float(sum(filler_features['cop_obj'])) / num_fillers

        kwargs['per_indefinite_np'] = \
            float(sum(filler_features['indefinite_np'])) / num_fillers
        kwargs['per_definite_np'] = \
            float(sum(filler_features['definite_np'])) / num_fillers
        kwargs['per_quantified_np'] = \
            float(sum(filler_features['quantified_np'])) / num_fillers

        kwargs['per_sentential_subj'] = \
            float(sum(filler_features['sentential_subj'])) / num_fillers

        kwargs['num_arg_to_other_pred'] = \
            sum(filler_features['arg_to_other_pred'])
        kwargs['per_arg_to_other_pred'] = \
            float(sum(filler_features['arg_to_other_pred'])) / num_fillers

        # TODO: should it be concatenated with dash or just list?
        f_head_ver_p_iarg_list = \
            ['-'.join([head_lemma, v_pred, iarg_label]) for head_lemma
             in filler_features['head_lemma']]
        if use_list:
            kwargs['f_head_ver_p_iarg'] = f_head_ver_p_iarg_list
        else:
            kwargs['f_head_ver_p_iarg'] = '-'.join(f_head_ver_p_iarg_list)

        f_head_synset_ver_p_iarg_list = \
            ['-'.join([head_synset_id, v_pred, iarg_label]) for head_synset_id
             in filler_features['head_synset_id'] if head_synset_id]
        if use_list:
            kwargs['f_head_synset_ver_p_iarg'] = f_head_synset_ver_p_iarg_list
        else:
            kwargs['f_head_synset_ver_p_iarg'] = \
                '-'.join(f_head_synset_ver_p_iarg_list)

        # TODO: should the fillers without predicates just be dropped?
        pf_argf_p_iarg_list = \
            ['-'.join([pred_f_lemma, arg_f_label, n_pred, iarg_label])
             for pred_f_lemma, arg_f_label
             in zip(
                filler_features['pred_f_lemma'],
                filler_features['arg_f_label'])
             if pred_f_lemma and arg_f_label]
        if use_list:
            kwargs['pf_argf_p_iarg'] = pf_argf_p_iarg_list
        else:
            kwargs['pf_argf_p_iarg'] = '-'.join(pf_argf_p_iarg_list)

        p_synset_id = helper.nominal_predicate_synset_mapping[n_pred]
        pf_synset_argf_p_synset_iarg_list = \
            ['-'.join([pred_f_synset_id, arg_f_label, p_synset_id, iarg_label])
             for pred_f_synset_id, arg_f_label
             in zip(
                filler_features['pred_f_synset_id'],
                filler_features['arg_f_label'])
             if pred_f_synset_id and arg_f_label]
        if use_list:
            kwargs['pf_synset_argf_p_synset_iarg'] = \
                pf_synset_argf_p_synset_iarg_list
        else:
            kwargs['pf_synset_argf_p_synset_iarg'] = \
                '-'.join(pf_synset_argf_p_synset_iarg_list)

        kwargs['argf_iarg_same?'] = \
            1 if iarg_label in filler_features['arg_f_label'] else 0

        # p_synset = wn.synset(p_synset_id)
        # wup_sim_list = []
        # for pred_f_synset_id in filler_features['pred_f_synset_id']:
        #     if pred_f_synset_id:
        #         pred_f_synset = wn.synset(pred_f_synset_id)
        #         wup_sim_list.append(p_synset.wup_similarity(pred_f_synset))
        # if wup_sim_list:
        #     kwargs['min_wup_sim_pf_p'] = min(wup_sim_list)
        # else:
        #     kwargs['min_wup_sim_pf_p'] = 0.0

        return cls(**kwargs)
