from os.path import join

from config import cfg

# mapping from nominal predicates to verbal predicates in imp_arg dataset
nominal_predicate_mapping = {
    'bid': 'bid',
    'sale': 'sell',
    'loan': 'loan',
    'cost': 'cost',
    'plan': 'plan',
    'investor': 'invest',
    'price': 'price',
    'loss': 'lose',
    'investment': 'invest',
    'fund': 'fund',
}

# list of core argument labels
core_arg_list = {'arg0', 'arg1', 'arg2', 'arg3', 'arg4'}

# mapping from nombank argument labels to syntactic labels
# for the 10 nominal predicates in imp_arg dataset
predicate_core_arg_mapping = {
    'bid': {
        'arg0': 'SUBJ',
        'arg1': 'PREP_for',
        'arg2': 'OBJ'
    },
    'sell': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_to',
        'arg3': 'PREP_for',
        'arg4': 'PREP'
    },
    'loan': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_to',
        'arg3': 'PREP',
        'arg4': 'PREP_at'
    },
    'cost': {
        'arg1': 'SUBJ',
        'arg2': 'OBJ',
        'arg3': 'PREP_to',
        'arg4': 'PREP'
    },
    'plan': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_for',
        'arg3': 'PREP_for'
    },
    'invest': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_in'
    },
    'price': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_at',
        'arg3': 'PREP'
    },
    'lose': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_to',
        'arg3': 'PREP_on'
    },
    'fund': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP',
        'arg3': 'PREP'
    },
}

# mapping from nombank function tags to function tags used in imp_arg dataset
nombank_function_tag_mapping = {
    'TMP': 'temporal',
    'LOC': 'location',
    'MNR': 'manner',
    'PNC': 'purpose',
    'NEG': 'negation',
    'EXT': 'extent',
    'ADV': 'adverbial',
    # tags below do not appear in implicit argument dataset
    'DIR': 'directional',
    'PRD': 'predicative',
    'CAU': 'cause',
    'DIS': 'discourse',
    'REF': 'reference'
}

# default path to store CoreNLP mapping
corenlp_mapping_path = join(cfg.data_path, 'imp_arg', 'corenlp_mapping.pkl')

# default path to store predicate mapping
predicate_mapping_path = join(cfg.data_path, 'imp_arg', 'predicate_mapping.pkl')

# default path to store candidate dictionary
candidate_dict_path = join(cfg.data_path, 'imp_arg', 'candidate_dict.pkl')

# default path to store all propositions
propositions_path = join(cfg.data_path, 'imp_arg', 'propositions.pkl')


def expand_wsj_fileid(fileid, ext=''):
    assert fileid[:4] == 'wsj_' and fileid[4:].isdigit()
    return fileid.split('_')[1][:2] + '/' + fileid + ext


def shorten_wsj_fileid(fileid):
    result = fileid[3:11]
    assert result[:4] == 'wsj_' and result[4:].isdigit()
    return result


def convert_nombank_label(label):
    if label[:3] == 'ARG':
        if label[3].isdigit():
            return label[:4].lower()
        elif label[3] == 'M':
            function_tag = label.split('-')[1]
            return nombank_function_tag_mapping.get(function_tag, '')
    return ''
