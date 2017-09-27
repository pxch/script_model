import urllib
from os.path import basename, exists, join
from zipfile import ZipFile

from config import cfg
from imp_arg_reader import ImplicitArgumentReader

data_path = cfg.data_path

imp_arg_dataset_name = 'implicit_argument_annotations.xml'
imp_arg_dataset_path = join(data_path, imp_arg_dataset_name)

imp_arg_dataset_url = \
    'http://lair.cse.msu.edu/projects/implicit_argument_annotations.zip'


def download_imp_arg_dataset():
    # download dataset from url
    print '\nDownloading implicit argument dataset from {}'.format(
        imp_arg_dataset_url)
    url_opener = urllib.URLopener()
    dataset_filename = basename(imp_arg_dataset_url)
    dataset_local_path = join(data_path, dataset_filename)
    url_opener.retrieve(imp_arg_dataset_url, dataset_local_path)

    # unzip dataset
    print '\nExtracting implicit argument dataset to {}'.format(data_path)
    dataset_zip = ZipFile(dataset_local_path)
    dataset_zip.extractall(data_path)
    dataset_zip.close()


def load_imp_arg_dataset(n_splits=10):
    if not exists(imp_arg_dataset_path):
        download_imp_arg_dataset()
    imp_arg_reader = ImplicitArgumentReader.from_dataset(
        imp_arg_dataset_path, n_splits=n_splits)
    return imp_arg_reader
