from operator import itemgetter

from common.imp_arg import helper

predicate_core_arg_mapping = helper.predicate_core_arg_mapping

nominal_predicate_synset_mapping = {
    'bid': 'bid.n.03',
    'sale': 'sale.n.01',
    'loan': 'loan.n.01',
    'cost': 'cost.n.01',
    'plan': 'plan.n.01',
    'investor': 'investor.n.01',
    'price': 'price.n.02',
    'loss': 'loss.n.02',
    'investment': 'investment.n.02',
    'fund': 'fund.n.01',
}

nominal_predicate_suffix_mapping = {
    'bid': '',
    'sale': '',
    'loan': '',
    'cost': '',
    'plan': '',
    'investor': '-or',
    'price': '',
    'loss': '',
    'investment': '-ment',
    'fund': '',
}

quantifier_list = [
    # both count and non-count nouns
    'all', 'some', 'any', 'enough', 'more', 'less', 'most', 'least',
    'a lot of', 'lots of', 'plenty of', 'a lack of',
    # only count nouns
    'many', 'few', 'several', 'a couple of', 'none',
    'both', 'each', 'either', 'neither',
    # only non-count nouns
    'much', 'little', 'a bit of', 'no', 'a great deal of', 'a good deal of',
]


indefinite_determiner_list = ['a', 'an']

definite_determiner_list = [
    'the', 'this', 'that',
    'which', 'what', 'whichever', 'whatever',
    # 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'whose',
]


def get_treepos(tree, wordnum):
    return tree.leaf_treeposition(wordnum)[:-1]


def get_token_range(tree, treepos):
    if type(treepos) != tuple:
        treepos = tuple(treepos)

    idx_list = []
    for idx in range(len(tree.leaves())):
        if tree.leaf_treeposition(idx)[:len(treepos)] == treepos:
            idx_list.append(idx)
    return idx_list


def get_left_sibling(treepos):
    left_treepos = list(treepos)
    found = False
    for idx in reversed(range(len(left_treepos))):
        if left_treepos[idx] > 0:
            left_treepos[idx] -= 1
            left_treepos = left_treepos[:idx + 1]
            found = True
            break
    if found:
        return tuple(left_treepos)
    else:
        return None


def get_right_sibling(tree, treepos):
    right_treepos = list(treepos)
    found = False
    for idx in reversed(range(len(right_treepos))):
        if len(tree[right_treepos[:idx]]) > right_treepos[idx] + 1:
            right_treepos[idx] += 1
            right_treepos = right_treepos[:idx + 1]
            found = True
            break
    if found:
        return tuple(right_treepos)
    else:
        return None


def get_parent(tree, treepos):
    child_token_range = get_token_range(tree, treepos)

    parent_treepos = list(treepos)
    while get_token_range(tree, parent_treepos) == child_token_range:
        parent_treepos = parent_treepos[:-1]

    if parent_treepos:
        return tuple(parent_treepos)
    else:
        return None


def get_syn_tree_path(dep_graph, token_idx_1, token_idx_2, msg_prefix=''):
    root_path_1 = dep_graph.get_root_path(token_idx_1, msg_prefix)[:-1]
    parent_idx_list_1 = map(itemgetter(1), root_path_1)

    root_path_2 = dep_graph.get_root_path(token_idx_2, msg_prefix)[:-1]
    parent_idx_list_2 = map(itemgetter(1), root_path_2)

    lowest_common_parent_idx = -1
    for parent_idx in parent_idx_list_1:
        if parent_idx in parent_idx_list_2:
            lowest_common_parent_idx = parent_idx
            break

    label_list_1 = []
    for label, parent_idx in root_path_1:
        label_list_1.append(label)
        if parent_idx == lowest_common_parent_idx:
            break

    label_list_2 = []
    for label, parent_idx in root_path_2:
        label_list_2.append(label)
        if parent_idx == lowest_common_parent_idx:
            break

    tree_path = label_list_1
    for label in reversed(label_list_2):
        tree_path.append(label)

    return tree_path
