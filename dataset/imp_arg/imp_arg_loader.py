import urllib
from os.path import basename, join
from zipfile import ZipFile

from config import cfg
from imp_arg_instance import ImplicitArgumentInstance
from utils import log

data_path = cfg.data_path

imp_arg_dataset_name = 'implicit_argument_annotations.xml'
imp_arg_dataset_path = join(data_path, imp_arg_dataset_name)

imp_arg_dataset_url = \
    'http://lair.cse.msu.edu/projects/implicit_argument_annotations.zip'


def download_dataset():
    # download dataset from url
    log.info('Downloading implicit argument dataset from {}'.format(
        imp_arg_dataset_url))
    url_opener = urllib.URLopener()
    dataset_filename = basename(imp_arg_dataset_url)
    dataset_local_path = join(data_path, dataset_filename)
    url_opener.retrieve(imp_arg_dataset_url, dataset_local_path)

    # unzip dataset
    log.info('Extracting implicit argument dataset to {}'.format(data_path))
    dataset_zip = ZipFile(dataset_local_path)
    dataset_zip.extractall(data_path)
    dataset_zip.close()


def read_dataset(file_path=imp_arg_dataset_path):
    log.info('Reading implicit argument dataset from {}'.format(file_path))

    all_instances = []
    input_xml = open(file_path, 'r')
    for line in input_xml.readlines()[1:-1]:
        instance = ImplicitArgumentInstance.parse(line.strip())
        all_instances.append(instance)
    input_xml.close()

    log.info('Found {} instances'.format(len(all_instances)))
    return all_instances


def write_dataset(all_instances, file_path):
    log.info('Writing implicit argument dataset to {}'.format(file_path))
    fout = open(file_path, 'w')

    fout.write('<annotations>\n')
    for instance in all_instances:
        fout.write(str(instance) + '\n')
    fout.write('</annotations>\n')

    fout.close()
