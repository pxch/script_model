from os.path import basename, splitext

from lxml import etree

from common.corenlp.document import Document
from dataset.corenlp.corenlp_target import CoreNLPTarget
from utils import get_console_logger, smart_file_handler

log = get_console_logger()


def read_doc_from_corenlp(filename):
    log.info('Reading CoreNLP document from {}'.format(filename))
    input_xml = smart_file_handler(filename)

    xml_parser = etree.XMLParser(target=CoreNLPTarget())
    sents, corefs = etree.parse(input_xml, xml_parser)
    doc_name = splitext(basename(filename))[0]
    doc = Document.construct(doc_name, sents, corefs)

    input_xml.close()

    return doc
