from copy import deepcopy
from os.path import join

from on.corpora import coreference, name, subcorpus

from common.corenlp import *
from config import cfg
from utils import check_type, convert_ontonotes_ner_tag, get_console_logger

log = get_console_logger()

ontonotes_annotations_source = join(cfg.ontonotes_root, 'english/annotations')


def read_coref_link(coref_link):
    check_type(coref_link, coreference.coreference_link)

    mention = Mention(
        coref_link.sentence_index,
        coref_link.start_word_index,
        coref_link.end_word_index + 1)
    if coref_link.subtree is not None:
        mention.text = coref_link.subtree.get_trace_adjusted_word_string()
    return mention


def read_coref_chain(coref_idx, coref_chain):
    check_type(coref_chain, coreference.coreference_chain)

    coref = Coreference(coref_idx)
    for coref_link in coref_chain:
        coref.add_mention(read_coref_link(coref_link))
    return coref


def read_coref_doc(coref_doc):
    check_type(coref_doc, coreference.coreference_document)

    all_corefs = []
    coref_idx = 0
    for coref_chain in coref_doc:
        if coref_chain.type == 'IDENT':
            coref = read_coref_chain(coref_idx, coref_chain)
            all_corefs.append(deepcopy(coref))
            coref_idx += 1
    return all_corefs


def read_name_doc(name_doc):
    check_type(name_doc, name.name_tagged_document)

    all_name_entities = []
    for name_entity_set in name_doc:
        for name_entity_hash in name_entity_set:
            for name_entity in name_entity_hash:
                all_name_entities.append(name_entity)
    return all_name_entities


def add_name_entity_to_doc(doc, name_entity):
    check_type(doc, Document)
    check_type(name_entity, name.name_entity)

    sent = doc.get_sent(name_entity.sentence_index)
    for token_idx in range(
            name_entity.start_word_index, name_entity.end_word_index + 1):
        token = sent.get_token(token_idx)
        # map ontonotes ner tags to coarse grained ner tags
        token.ner = convert_ontonotes_ner_tag(name_entity.type)


def read_conll_depparse(input_path):
    fin = open(input_path, 'r')

    all_sents = []
    sent_idx = 0
    sent = Sentence(sent_idx)

    for line_idx, line in enumerate(fin.readlines()):
        if line == '\n':
            all_sents.append(deepcopy(sent))
            sent_idx += 1
            sent = Sentence(sent_idx)
        else:
            items = line.strip().split('\t')
            try:
                token_idx = int(items[0])
            except ValueError:
                continue
            if token_idx == sent.num_tokens:
                log.warn(
                    'line #{} ({}) has duplicated token index, ignored.'.format(
                        line_idx, line.strip().replace('\t', ' ')))
                continue
            word = items[1]
            lemma = items[2]
            pos = items[4]
            sent.add_token(Token(word, lemma, pos))
            try:
                head_idx = int(items[6])
            except ValueError:
                continue
            dep_label = items[7]
            if dep_label != 'root':
                sent.add_dep(Dependency(
                    label=dep_label,
                    head_idx=head_idx - 1,
                    mod_idx=token_idx - 1,
                    extra=False))
            if items[8] != '_':
                for e_dep in items[8].strip().split('|'):
                    try:
                        e_dep_head_idx = int(e_dep.split(':')[0])
                    except ValueError:
                        continue
                    e_dep_label = ':'.join(e_dep.split(':')[1:])
                    sent.add_dep(Dependency(
                        label=e_dep_label,
                        head_idx=e_dep_head_idx - 1,
                        mod_idx=token_idx - 1,
                        extra=True))

    return all_sents


def read_doc_from_ontonotes(coref_doc, name_doc):
    doc_id = coref_doc.document_id.split('@')[0]
    assert doc_id == name_doc.document_id.split('@')[0], \
        '{} and {} do not have the same document_id'.format(coref_doc, name_doc)

    log.info('Reading ontonotes document {}'.format(doc_id))

    conll_file_path = join(ontonotes_annotations_source, doc_id + '.depparse')

    all_sents = read_conll_depparse(conll_file_path)

    all_corefs = read_coref_doc(coref_doc)

    doc_name = doc_id.split('/')[-1]
    doc = Document.construct(doc_name, all_sents, all_corefs)

    for name_entity in read_name_doc(name_doc):
        add_name_entity_to_doc(doc, name_entity)

    return doc


def read_all_docs_from_ontonotes(corpora):
    check_type(corpora, subcorpus)
    assert 'coref' in corpora, 'corpora must contains coref bank'
    assert 'name' in corpora, 'corpora must contains name bank'

    all_docs = []
    for coref_doc, name_doc in zip(corpora['coref'], corpora['name']):
        doc = read_doc_from_ontonotes(coref_doc, name_doc)
        all_docs.append(doc)

    return all_docs
